"""Train a BPE tokenizer on a pre-built sentence-pair corpus.

The resulting ``tokenizer.json`` is compatible with ``BPEVocab`` in
:mod:`dataset.vocab`. Special tokens are reserved at fixed positions 0–4:

    0 <pad>   1 <unk>   2 <eos>   3 <sos>   4 <mask>

so the rest of the project keeps working (``padding_idx=0``,
``ignore_index=0``) without any model-side change.

Usage::

    python -m dataset.build_tokenizer \
        --corpus dataset/data/corpus/train.txt \
        --out dataset/data/tokenizer/tokenizer.json \
        --vocab_size 30000
"""

import argparse
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from .vocab import SPECIAL_TOKENS, BPEVocab


DEFAULT_CORPUS = Path(__file__).resolve().parent / "data" / "corpus" / "train.txt"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "tokenizer" / "tokenizer.json"


def _corpus_lines(corpus_path: Path):
    """Stream each sentence (both halves of each pair) to the trainer.

    The trainer reads an iterator of strings, so we yield ``sentence1`` and
    ``sentence2`` separately to keep memory flat even for a multi-GB file.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                a, b = line.split("\t", 1)
                if a:
                    yield a
                if b:
                    yield b
            else:
                yield line


def build_tokenizer(corpus_paths: List[Path], out_path: Path,
                    vocab_size: int = 30000, min_frequency: int = 2,
                    lowercase: bool = True) -> BPEVocab:
    """Train a Byte-Level BPE tokenizer, save it, return a ``BPEVocab``.

    We use GPT-2 / RoBERTa-style Byte-Level BPE because:
      * whitespace is preserved through the byte-level alphabet, so
        ``decode(encode(s)) == s`` (modulo case normalization);
      * there is no need for an explicit ``<unk>`` at inference — any
        byte sequence can be expressed with the 256-byte base alphabet;
      * encoding speed from the Rust backend is very competitive.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    norm_steps = [normalizers.NFD(), normalizers.StripAccents()]
    if lowercase:
        norm_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(norm_steps)

    # ByteLevel pre-tokenizer maps every Unicode byte to a visible char from a
    # fixed 256-symbol alphabet, which means whitespace and punctuation are
    # always representable and never collapse.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # Order here pins special tokens to ids 0..4.
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def _iter_all():
        for path in corpus_paths:
            yield from _corpus_lines(path)

    tokenizer.train_from_iterator(_iter_all(), trainer=trainer)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))

    vocab = BPEVocab(tokenizer)
    print(f"Saved tokenizer to {out_path} (vocab size = {len(vocab)})")
    return vocab


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, nargs="+", default=[DEFAULT_CORPUS],
                   help="One or more corpus files to train on")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output path for tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--no_lowercase", action="store_true",
                   help="Keep original case (default: lowercase everything)")
    return p.parse_args()


def main():
    args = parse_args()
    for path in args.corpus:
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
    build_tokenizer(
        args.corpus, args.out,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=not args.no_lowercase,
    )


if __name__ == "__main__":
    main()
