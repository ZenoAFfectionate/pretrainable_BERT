"""SQuAD v1.1 data loading, featurization, and official EM/F1 metrics.

This module is shared by ``finetune.py`` and ``evaluate.py`` so both scripts
see exactly the same feature layout and metric implementation.

Features assume a single window per example (no doc-stride): the question is
truncated to ``max_query_len`` tokens, the context is right-truncated to fit
in the remaining budget, and padding is filled with ``<pad>``. When building
training features, we drop examples whose gold answer falls outside the
retained context window.
"""

import collections
import json
import re
import string
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .vocab import BPEVocab


class SquadExample:
    """One raw SQuAD question, retained for post-processing."""

    __slots__ = ("qid", "question", "context", "answers", "answer_start")

    def __init__(self, qid, question, context, answers, answer_start):
        self.qid = qid
        self.question = question
        self.context = context
        self.answers = answers            # List[str]  all gold texts
        self.answer_start = answer_start  # int | None first-answer char offset


def load_squad_examples(path: str) -> List[SquadExample]:
    """Load a SQuAD v1.1 JSON file into flat SquadExample objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    examples: List[SquadExample] = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                answers = [a["text"] for a in qa.get("answers", [])]
                ans_start = qa["answers"][0]["answer_start"] if qa.get("answers") else None
                examples.append(SquadExample(
                    qid=qa["id"],
                    question=qa["question"],
                    context=ctx,
                    answers=answers,
                    answer_start=ans_start,
                ))
    return examples


class SquadFeature:
    """Tokenized feature ready for the model."""

    __slots__ = (
        "input_ids", "segment_ids",
        "start_position", "end_position",
        "example_index", "context_token_offsets", "context_start_in_input",
    )

    def __init__(self, input_ids, segment_ids, start_position, end_position,
                 example_index, context_token_offsets, context_start_in_input):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.example_index = example_index
        # Only retained for eval features — saves RAM during large-scale training.
        self.context_token_offsets = context_token_offsets
        self.context_start_in_input = context_start_in_input


def convert_examples_to_features(examples: List[SquadExample], vocab: BPEVocab,
                                 seq_len: int, is_training: bool,
                                 max_query_len: int = 64,
                                 logger=None) -> List[SquadFeature]:
    """Tokenize SquadExamples into fixed-length SquadFeatures."""
    pad_id = vocab.pad_index
    sos_id = vocab.sos_index
    eos_id = vocab.eos_index
    tokenizer = vocab.tokenizer

    features: List[SquadFeature] = []
    dropped_no_answer = 0
    dropped_overflow = 0

    for ex_idx, ex in enumerate(examples):
        q_enc = tokenizer.encode(ex.question, add_special_tokens=False)
        q_ids = q_enc.ids[:max_query_len]

        # 3 specials: [SOS] + 2 x [EOS]
        budget = seq_len - 3 - len(q_ids)
        if budget <= 0:
            dropped_overflow += 1
            continue

        c_enc = tokenizer.encode(ex.context, add_special_tokens=False)
        c_ids = c_enc.ids
        c_offsets = c_enc.offsets  # list of (start_char, end_char) per sub-word

        if len(c_ids) > budget:
            c_ids = c_ids[:budget]
            c_offsets = c_offsets[:budget]

        context_start_in_input = 1 + len(q_ids) + 1
        input_ids = [pad_id] * seq_len
        segment_ids = [0] * seq_len

        input_ids[0] = sos_id
        segment_ids[0] = 1
        for i, tid in enumerate(q_ids):
            input_ids[1 + i] = tid
            segment_ids[1 + i] = 1
        input_ids[context_start_in_input - 1] = eos_id
        segment_ids[context_start_in_input - 1] = 1

        for i, tid in enumerate(c_ids):
            input_ids[context_start_in_input + i] = tid
            segment_ids[context_start_in_input + i] = 2
        input_ids[context_start_in_input + len(c_ids)] = eos_id
        segment_ids[context_start_in_input + len(c_ids)] = 2

        start_position = 0
        end_position = 0
        if is_training:
            if not ex.answers or ex.answer_start is None:
                dropped_no_answer += 1
                continue
            gold_text = ex.answers[0]
            a_start_char = ex.answer_start
            a_end_char = a_start_char + len(gold_text)

            tok_start = tok_end = None
            for i, (s, e) in enumerate(c_offsets):
                if s <= a_start_char < e and tok_start is None:
                    tok_start = i
                if s < a_end_char <= e:
                    tok_end = i
                    break
            if tok_start is None or tok_end is None or tok_end < tok_start:
                dropped_overflow += 1
                continue

            start_position = context_start_in_input + tok_start
            end_position = context_start_in_input + tok_end

        features.append(SquadFeature(
            input_ids=input_ids,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            example_index=ex_idx,
            context_token_offsets=c_offsets if not is_training else None,
            context_start_in_input=context_start_in_input,
        ))

    if logger is not None:
        split = "train" if is_training else "eval"
        logger.info(
            "Built %d %s features from %d examples (dropped: no_answer=%d, overflow=%d)",
            len(features), split, len(examples), dropped_no_answer, dropped_overflow,
        )
    return features


class SquadFeatureDataset(Dataset):
    """Thin torch.utils.data.Dataset adapter over a list of SquadFeature."""

    def __init__(self, features: List[SquadFeature], is_training: bool):
        self.features = features
        self.is_training = is_training

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        out = {
            "input_ids": torch.tensor(f.input_ids, dtype=torch.long),
            "segment_ids": torch.tensor(f.segment_ids, dtype=torch.long),
        }
        if self.is_training:
            out["start_position"] = torch.tensor(f.start_position, dtype=torch.long)
            out["end_position"] = torch.tensor(f.end_position, dtype=torch.long)
        else:
            out["feature_index"] = torch.tensor(idx, dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# Official SQuAD v1.1 metric (normalization + EM + F1)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation + articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def _em_score(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def squad_metrics(predictions: Dict[str, str],
                  examples: List[SquadExample]) -> Dict[str, float]:
    """EM / F1 computed as the max over gold answers per question."""
    em_total = 0.0
    f1_total = 0.0
    n = 0
    for ex in examples:
        if ex.qid not in predictions:
            continue
        n += 1
        pred = predictions[ex.qid]
        golds = ex.answers if ex.answers else [""]
        em_total += max(_em_score(pred, g) for g in golds)
        f1_total += max(_f1_score(pred, g) for g in golds)
    if n == 0:
        return {"exact_match": 0.0, "f1": 0.0, "count": 0}
    return {"exact_match": 100.0 * em_total / n,
            "f1": 100.0 * f1_total / n, "count": n}
