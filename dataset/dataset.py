"""BERT pre-training dataset.

Produces samples with Next Sentence Prediction (NSP) + Masked Language Model
(MLM) signals. The corpus format is one sentence pair per line separated by a
tab: ``sentence_1\\tsentence_2``.

Design choices versus a naive implementation
--------------------------------------------
* **BPE sub-word tokenization.** The vocabulary is a HuggingFace BPE tokenizer
  (see :class:`dataset.vocab.BPEVocab`), so OOV words are split into known
  pieces instead of collapsing to ``<unk>``.
* **Offset-based indexing.** Instead of loading all tokenized pairs into RAM,
  we store byte offsets of valid lines and tokenise on-the-fly in
  ``__getitem__``. This keeps memory usage at ~1 GB even for 100M+ pairs.
* **Proper BERT truncation.** We reserve 3 slots for ``[CLS]`` + two
  ``[SEP]`` tokens and truncate the longer of the two sentences one token at
  a time until the pair fits, instead of lopping off the tail and potentially
  erasing sentence 2 entirely.
* **Pre-allocated padded tensors.** We fill ``torch.long`` tensors of the
  target length directly instead of building Python lists and padding with
  list ``extend``.
* **MLM random replacement skips special tokens** (pad/unk/sos/eos/mask) so
  the "replace with random vocab" branch never substitutes a reserved symbol.
"""

import array
import random
from typing import List, Tuple

import torch
import tqdm
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    """BERT pre-training dataset backed by offset-based on-demand tokenization.

    Parameters
    ----------
    corpus_path : str
        Path to a file with one ``sentence1\\tsentence2`` pair per line.
    vocab : Vocab
        Shared vocabulary object. Must expose ``stoi``/``itos`` dicts and the
        special indices (``pad_index``, ``sos_index``, ``eos_index``,
        ``mask_index``, ``unk_index``).
    seq_len : int
        Fixed output sequence length. Inputs longer than ``seq_len - 3`` are
        truncated; shorter inputs are right-padded with ``pad_index``.
    encoding : str
        File encoding.
    mask_prob : float
        Probability of choosing a token to be included in the MLM objective.
    """

    def __init__(self, corpus_path: str, vocab, seq_len: int,
                 encoding: str = "utf-8", mask_prob: float = 0.15):
        self.vocab = vocab
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self._corpus_path = corpus_path
        self._encoding = encoding

        # Cache special token ids for fast access in the hot path.
        self.pad_index = vocab.pad_index
        self.unk_index = vocab.unk_index
        self.sos_index = vocab.sos_index
        self.eos_index = vocab.eos_index
        self.mask_index = vocab.mask_index
        self._special_ids = vocab.special_indices
        self._vocab_size = len(vocab)
        # Precompute the set of non-special ids for the 10% random-replacement
        # branch; falling back to `random.randrange(V)` and re-rolling would
        # be wasteful when specials are a tiny fraction of the vocab.
        non_special = [i for i in range(self._vocab_size) if i not in self._special_ids]
        self._non_special_ids = non_special

        # Index valid line byte-offsets (NOT the full tokenized content).
        # Using array('Q') = 8 bytes per offset → ~1.1 GB for 139M pairs.
        self._offsets = self._index_valid_lines(corpus_path, encoding)
        self._file = None  # Lazily opened per DataLoader worker

        if not self._offsets:
            raise ValueError(f"No valid sentence pairs found in {corpus_path}")

    # ------------------------------------------------------------------ I/O

    def _index_valid_lines(self, path: str, encoding: str):
        """Scan the corpus and record byte offsets of valid lines.

        Uses binary-mode reads so ``f.tell()`` returns reliable byte offsets
        for later seeking. Stores offsets in a compact ``array('Q')``
        (unsigned 64-bit ints) instead of Python lists.
        """
        offsets = array.array('Q')
        with open(path, "rb") as f:
            offset = 0
            for raw in tqdm.tqdm(f, desc=f"Indexing {path}"):
                try:
                    line = raw.decode(encoding).strip()
                except UnicodeDecodeError:
                    offset = f.tell()
                    continue
                if not line or "\t" not in line:
                    offset = f.tell()
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    offset = f.tell()
                    continue
                s1, s2 = parts[0].strip(), parts[1].strip()
                if s1 and s2:
                    offsets.append(offset)
                offset = f.tell()
        return offsets

    def _get_file(self):
        """Lazily open the corpus file — one handle per DataLoader worker."""
        if self._file is None:
            self._file = open(self._corpus_path, "rb")
        return self._file

    def _read_pair(self, index: int) -> Tuple[List[int], List[int]]:
        """Seek to *index*-th pair and return tokenized (t1_ids, t2_ids)."""
        offset = self._offsets[index]
        f = self._get_file()
        f.seek(offset)
        raw = f.readline().decode(self._encoding).strip()
        parts = raw.split("\t")
        s1 = parts[0].strip()
        s2 = parts[1].strip()
        return self.vocab.encode(s1), self.vocab.encode(s2)

    # ------------------------------------------------------------- sampling

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index: int):
        t1_ids, t2_ids = self._read_pair(index)

        # NSP: 50% same pair (is_next=1), 50% replace t2 with a random one.
        if random.random() < 0.5:
            is_next = 1
        else:
            t2_ids = self._random_pair_second(exclude=index)
            is_next = 0

        # MLM: mask tokens in each sentence independently.
        t1_input, t1_label = self._mask_tokens(t1_ids)
        t2_input, t2_label = self._mask_tokens(t2_ids)

        # Truncate so that [CLS] + t1 + [SEP] + t2 + [SEP] fits in seq_len.
        t1_input, t1_label, t2_input, t2_label = self._truncate_pair(
            t1_input, t1_label, t2_input, t2_label,
        )

        seq_len = self.seq_len
        pad = self.pad_index

        # Pre-allocate fixed-length tensors (faster than list build + pad).
        bert_input = torch.full((seq_len,), pad, dtype=torch.long)
        bert_label = torch.full((seq_len,), pad, dtype=torch.long)
        segment_label = torch.zeros(seq_len, dtype=torch.long)

        # Segment A: [CLS] + t1 + [SEP]
        a_len = 1 + len(t1_input) + 1
        bert_input[0] = self.sos_index
        bert_input[1:1 + len(t1_input)] = torch.tensor(t1_input, dtype=torch.long)
        bert_input[a_len - 1] = self.eos_index
        bert_label[1:1 + len(t1_label)] = torch.tensor(t1_label, dtype=torch.long)
        segment_label[:a_len] = 1

        # Segment B: t2 + [SEP]
        b_len = len(t2_input) + 1
        bert_input[a_len:a_len + len(t2_input)] = torch.tensor(t2_input, dtype=torch.long)
        bert_input[a_len + b_len - 1] = self.eos_index
        bert_label[a_len:a_len + len(t2_label)] = torch.tensor(t2_label, dtype=torch.long)
        segment_label[a_len:a_len + b_len] = 2

        return {
            "bert_input": bert_input,
            "bert_label": bert_label,
            "segment_label": segment_label,
            "is_next": torch.tensor(is_next, dtype=torch.long),
        }

    # -------------------------------------------------------------- helpers

    def _random_pair_second(self, exclude: int) -> List[int]:
        """Return the second sentence of a random pair that isn't ``exclude``."""
        n = len(self._offsets)
        if n <= 1:
            return self._read_pair(0)[1]
        while True:
            j = random.randrange(n)
            if j != exclude:
                return self._read_pair(j)[1]

    def _mask_tokens(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """Apply the BERT 15%/80-10-10 masking rule.

        Returns ``(input_ids, label_ids)`` where ``label_ids[i]`` is the original
        id for masked positions and ``pad_index`` (0) for unmasked positions so
        the loss with ``ignore_index=0`` only sees masked tokens.
        """
        input_ids = list(ids)
        labels = [self.pad_index] * len(ids)
        mask_prob = self.mask_prob

        for i, tok in enumerate(ids):
            if tok in self._special_ids:
                # Never mask specials; leave label as pad so it contributes no loss.
                continue
            if random.random() >= mask_prob:
                continue

            labels[i] = tok
            r = random.random()
            if r < 0.8:
                input_ids[i] = self.mask_index
            elif r < 0.9:
                input_ids[i] = random.choice(self._non_special_ids)
            # else: keep the original token (10% branch)

        return input_ids, labels

    def _truncate_pair(self, t1_input, t1_label, t2_input, t2_label):
        """Truncate the pair so 3 special tokens + content fits in seq_len.

        Follows the canonical BERT heuristic: pop one token from the **longer**
        of the two sequences until the combined length fits.
        """
        budget = self.seq_len - 3  # [CLS], [SEP]_A, [SEP]_B
        while len(t1_input) + len(t2_input) > budget:
            if len(t1_input) >= len(t2_input):
                t1_input.pop()
                t1_label.pop()
            else:
                t2_input.pop()
                t2_label.pop()
        return t1_input, t1_label, t2_input, t2_label
