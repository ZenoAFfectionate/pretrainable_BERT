"""BPE-based vocabulary used across the BERT pre-training pipeline.

``BPEVocab`` wraps a HuggingFace ``tokenizers.Tokenizer`` but keeps the same
lightweight API surface the rest of the project already expects:

    * attributes: ``pad_index``, ``unk_index``, ``eos_index``, ``sos_index``,
      ``mask_index``, ``stoi``, ``itos``, ``special_indices``
    * methods:   ``encode``, ``decode``, ``to_seq``, ``from_seq``,
                 ``save_vocab``, ``load_vocab``, ``__len__``

Special tokens are pinned to indices 0–4 regardless of frequency:

    0 <pad>   1 <unk>   2 <eos>   3 <sos>   4 <mask>

That keeps the model's ``SegmentEmbedding(padding_idx=0)`` and the dataset's
``CrossEntropyLoss(ignore_index=0)`` contracts unchanged when switching from
the old whitespace vocab to sub-word BPE.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from tokenizers import Tokenizer


# Reserved special tokens (order matters: index = position in this tuple).
SPECIAL_TOKENS: Sequence[str] = ("<pad>", "<unk>", "<eos>", "<sos>", "<mask>")

PAD_INDEX, UNK_INDEX, EOS_INDEX, SOS_INDEX, MASK_INDEX = range(len(SPECIAL_TOKENS))


class BPEVocab:
    """Sub-word vocabulary backed by a trained BPE tokenizer.

    Parameters
    ----------
    tokenizer : Tokenizer
        A trained ``tokenizers.Tokenizer`` instance whose vocabulary has the
        five special tokens at positions 0–4.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

        # Cache id mappings so look-ups in the dataset hot path are plain dict
        # reads rather than cross-boundary calls into the Rust tokenizer.
        self.stoi = dict(tokenizer.get_vocab())
        self.itos: List[str] = [""] * len(self.stoi)
        for tok, idx in self.stoi.items():
            self.itos[idx] = tok

        # Verify and expose the reserved specials.
        for expected, idx in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
            actual = self.itos[idx]
            if actual != expected:
                raise ValueError(
                    f"Special token mismatch at position {idx}: "
                    f"expected {expected!r}, got {actual!r}. "
                    "Re-train the tokenizer with the correct special token order."
                )

        self.pad_index = PAD_INDEX
        self.unk_index = UNK_INDEX
        self.eos_index = EOS_INDEX
        self.sos_index = SOS_INDEX
        self.mask_index = MASK_INDEX

    # --------------------------------------------------------------- meta

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def special_indices(self) -> set:
        return {self.pad_index, self.unk_index, self.eos_index,
                self.sos_index, self.mask_index}

    @property
    def tokenizer(self) -> Tokenizer:
        """Raw HuggingFace tokenizer (exposed for advanced use)."""
        return self._tokenizer

    # --------------------------------------------------------- encode/decode

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert a string to a list of sub-word ids."""
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens,
        ).ids

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        """Vectorized encode — use when tokenizing many lines at once."""
        batch = list(texts)
        if not batch:
            return []
        enc = self._tokenizer.encode_batch(batch, add_special_tokens=False)
        return [e.ids for e in enc]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of sub-word ids back to a string."""
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # -------------------------------------------- legacy Vocab-style helpers

    def to_seq(self, sentence: Union[str, List[str]], seq_len: Optional[int] = None,
               with_eos: bool = False, with_sos: bool = False,
               with_len: bool = False):
        """BPE-aware version of the old ``Vocab.to_seq``.

        If ``sentence`` is a list of whitespace-pre-split tokens, it is re-joined
        before BPE (BPE works on raw strings, not on pre-tokenized word lists).
        """
        if isinstance(sentence, list):
            sentence = " ".join(sentence)

        seq = self.encode(sentence)

        if with_eos:
            seq = seq + [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is not None:
            if len(seq) < seq_len:
                seq = seq + [self.pad_index] * (seq_len - len(seq))
            else:
                seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq: Sequence[int], join: bool = True,
                 with_pad: bool = False) -> Union[str, List[str]]:
        """Inverse of ``to_seq`` — mirrors the old ``Vocab.from_seq``.

        ``join=True`` returns a decoded string (recommended for BPE).
        ``join=False`` returns the list of raw sub-word strings.
        """
        if join:
            return self.decode(seq, skip_special_tokens=not with_pad)
        return [
            self.itos[idx] if 0 <= idx < len(self.itos) else f"<{idx}>"
            for idx in seq
            if with_pad or idx != self.pad_index
        ]

    # ------------------------------------------------------------ persistence

    def save_vocab(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
        return path

    @classmethod
    def load_vocab(cls, path: Union[str, Path]) -> "BPEVocab":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)


# Backward-compatible aliases so scripts that still import the old names keep
# working without code changes.
Vocab = BPEVocab
WordVocab = BPEVocab
