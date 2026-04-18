"""Evaluate a SQuAD-fine-tuned QA model on the dev split — no training.

Loads a checkpoint produced by ``finetune.py`` (a full :class:`QAModel` state
dict plus a ``config`` block), rebuilds the encoder + QA head, and reports
Exact-Match / F1 on the supplied dev JSON. Predictions for every qid are
written to ``eval_predictions.json`` inside the run directory.

    python evaluate.py \\
        --checkpoint result/<finetune_run>/checkpoints/best_model.pt \\
        --vocab_path dataset/data/tokenizer/tokenizer.json \\
        --dev_file   dataset/tune/squad/dev-v1.1.json
"""

import argparse
import json
import os
import warnings

import torch
from torch.utils.data import DataLoader

from dataset import (
    BPEVocab, SquadFeatureDataset,
    convert_examples_to_features, load_squad_examples,
)
from model import load_qa_model_from_checkpoint
from utils import TrainingLogger

from finetune import run_qa_evaluation  # reuse the prediction loop

warnings.filterwarnings("ignore")


def build_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Evaluate a fine-tuned QA model on SQuAD v1.1")

    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to a finetune checkpoint (best_model.pt)")
    parser.add_argument("--vocab_path", required=True, type=str,
                        help="Path to tokenizer.json")
    parser.add_argument("--dev_file", required=True, type=str,
                        help="Path to SQuAD dev-v1.1.json")

    parser.add_argument("--seq_len", type=int, default=None,
                        help="Override sequence length (default: use value from checkpoint)")
    parser.add_argument("--max_query_len", type=int, default=None,
                        help="Override max question length (default: use value from checkpoint)")
    parser.add_argument("--max_answer_tokens", type=int, default=None,
                        help="Override max answer tokens (default: use value from checkpoint)")

    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cuda_devices", type=str, default=None)

    parser.add_argument("--result_root", type=str, default="result")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--max_dev_examples", type=int, default=-1,
                        help="Cap on dev examples (-1 = all)")
    return parser


def run(args) -> None:
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    tlogger = TrainingLogger(result_root=args.result_root, run_name=args.run_name,
                             prefix="evaluate")
    tlogger.info("Starting SQuAD v1.1 evaluation (no training)")

    tlogger.info("Loading vocabulary from %s", args.vocab_path)
    vocab = BPEVocab.load_vocab(args.vocab_path)
    tlogger.info("Vocabulary size: %d", len(vocab))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlogger.info("Using device: %s", device)

    tlogger.info("Loading finetune checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {}) or {}
    seq_len = args.seq_len or cfg.get("seq_len", 384)
    max_query_len = args.max_query_len or cfg.get("max_query_len", 64)
    max_answer_tokens = args.max_answer_tokens or cfg.get("max_answer_tokens", 30)

    model, model_type = load_qa_model_from_checkpoint(
        ckpt, vocab_size=len(vocab), logger=tlogger,
    )
    model = model.to(device)
    model.eval()

    tlogger.info("Model: %s | seq_len=%d max_query_len=%d max_answer_tokens=%d",
                 model_type, seq_len, max_query_len, max_answer_tokens)

    tlogger.info("Loading SQuAD dev from %s", args.dev_file)
    dev_examples = load_squad_examples(args.dev_file)
    if args.max_dev_examples > 0:
        dev_examples = dev_examples[:args.max_dev_examples]
    tlogger.info("Loaded %d dev examples", len(dev_examples))

    dev_features = convert_examples_to_features(
        dev_examples, vocab, seq_len=seq_len, is_training=False,
        max_query_len=max_query_len, logger=tlogger,
    )
    dev_loader = DataLoader(
        SquadFeatureDataset(dev_features, is_training=False),
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    tlogger.save_config({
        "task": "squad_v1.1_evaluate",
        "model_type": model_type,
        "finetune_checkpoint": str(args.checkpoint),
        "finetune_config": cfg,
        "training_args": vars(args),
        "dev_features": len(dev_features),
    })

    metrics, predictions = run_qa_evaluation(
        model, dev_features, dev_examples, dev_loader, device,
        fp16=args.fp16, max_answer_tokens=max_answer_tokens,
    )
    tlogger.log_epoch(epoch=-1, split="dev",
                      exact_match=metrics["exact_match"],
                      f1=metrics["f1"], count=metrics["count"])
    tlogger.info("Dev metrics: EM=%.2f  F1=%.2f  (n=%d)",
                 metrics["exact_match"], metrics["f1"], metrics["count"])

    pred_path = tlogger.run_dir / "eval_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions},
                  f, indent=2, ensure_ascii=False)
    tlogger.info("Wrote predictions to %s", pred_path)
    tlogger.close()


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
