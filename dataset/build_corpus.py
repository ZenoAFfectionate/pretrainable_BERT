"""Build a BERT pre-training corpus from Wikipedia + BookCorpusOpen.

Flow:

    HF ``wikimedia/wikipedia`` (streaming)  ┐
    HF ``lucadiliello/bookcorpusopen``  (streaming)  ┘
                       │
                       ▼
      split into paragraphs (blank-line) then NLTK Punkt sentences
                       │
                       ▼
   emit adjacent ``sentence_i <TAB> sentence_{i+1}`` pairs
                       │
                       ▼
           dataset/data/corpus/{train,test}.txt

The output file format matches what ``BERTDataset`` consumes: one line per pair,
``sentence1\\tsentence2``. Adjacency within a paragraph preserves proper NSP
signal (unlike the old half-sentence hack).

Downloading the full Wikipedia snapshot is ~20 GB and takes hours; the script
caps the amount of data via ``--max_articles`` / ``--max_books``. Use ``-1`` to
mean "no limit".
"""

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List

import nltk
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "corpus"

WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
BOOKCORPUS_REPO = "lucadiliello/bookcorpusopen"

MIN_SENT_CHARS = 16    # drop very short sentences (e.g. section headers)
MIN_SENT_WORDS = 4


def _ensure_punkt() -> None:
    """Make sure NLTK's sentence tokenizer models are available."""
    # NLTK 3.9 switched the default resource name to 'punkt_tab'; older code
    # may still have 'punkt' cached. Ensure both if missing.
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    nltk.download("punkt_tab", quiet=True)


def _split_sentences(text: str) -> List[str]:
    """Paragraph-aware sentence splitting using NLTK Punkt."""
    if not text:
        return []
    sentences: List[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sent in nltk.sent_tokenize(paragraph):
            sent = " ".join(sent.split())  # collapse whitespace
            if len(sent) < MIN_SENT_CHARS:
                continue
            if len(sent.split()) < MIN_SENT_WORDS:
                continue
            sentences.append(sent)
        # Mark a paragraph break: we use an empty string to signal "do not
        # pair across this boundary" in :func:`_emit_pairs`.
        sentences.append("")
    return sentences


def _emit_pairs(sentences: Iterable[str], writer, test_writer,
                test_ratio: float, rng: random.Random) -> int:
    """Emit adjacent sentence pairs, skipping across paragraph breaks.

    Returns the number of pairs emitted.
    """
    count = 0
    prev = ""
    for sent in sentences:
        if sent == "":  # paragraph boundary
            prev = ""
            continue
        if prev:
            line = f"{prev}\t{sent}\n"
            if rng.random() < test_ratio:
                test_writer.write(line)
            else:
                writer.write(line)
            count += 1
        prev = sent
    return count


def build_from_wikipedia(out_train, out_test, max_articles: int,
                         test_ratio: float, rng: random.Random) -> int:
    total_pairs = 0
    print(f"Streaming {WIKIPEDIA_REPO} ({WIKIPEDIA_CONFIG}) ...")
    ds = load_dataset(WIKIPEDIA_REPO, WIKIPEDIA_CONFIG,
                      split="train", streaming=True)
    it = ds if max_articles < 0 else ds.take(max_articles)
    total = None if max_articles < 0 else max_articles
    for article in tqdm(it, total=total, desc="wikipedia"):
        sents = _split_sentences(article.get("text", ""))
        total_pairs += _emit_pairs(sents, out_train, out_test, test_ratio, rng)
    return total_pairs


def build_from_bookcorpus(out_train, out_test, max_books: int,
                          test_ratio: float, rng: random.Random) -> int:
    total_pairs = 0
    print(f"Streaming {BOOKCORPUS_REPO} ...")
    ds = load_dataset(BOOKCORPUS_REPO, split="train", streaming=True)
    it = ds if max_books < 0 else ds.take(max_books)
    total = None if max_books < 0 else max_books
    for book in tqdm(it, total=total, desc="bookcorpus"):
        sents = _split_sentences(book.get("text", ""))
        total_pairs += _emit_pairs(sents, out_train, out_test, test_ratio, rng)
    return total_pairs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory for train.txt / test.txt")
    p.add_argument("--max_articles", type=int, default=-1,
                   help="Max Wikipedia articles to process (-1 for all)")
    p.add_argument("--max_books", type=int, default=-1,
                   help="Max BookCorpus books to process (-1 for all)")
    p.add_argument("--test_ratio", type=float, default=0.005,
                   help="Fraction of pairs routed to test.txt (default 0.5%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_wiki", action="store_true")
    p.add_argument("--skip_books", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_punkt()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.txt"
    test_path = args.out_dir / "test.txt"

    rng = random.Random(args.seed)
    wiki_pairs = 0
    book_pairs = 0

    with open(train_path, "w", encoding="utf-8") as ftrain, \
         open(test_path, "w", encoding="utf-8") as ftest:
        if not args.skip_wiki:
            wiki_pairs = build_from_wikipedia(
                ftrain, ftest, args.max_articles, args.test_ratio, rng,
            )
        if not args.skip_books:
            book_pairs = build_from_bookcorpus(
                ftrain, ftest, args.max_books, args.test_ratio, rng,
            )

    total = wiki_pairs + book_pairs
    print(f"Wrote {total:,} pairs (wiki={wiki_pairs:,}, books={book_pairs:,}) "
          f"to {args.out_dir}")
    print(f"  train: {train_path} ({os.path.getsize(train_path):,} bytes)")
    print(f"  test:  {test_path}  ({os.path.getsize(test_path):,} bytes)")


if __name__ == "__main__":
    main()
