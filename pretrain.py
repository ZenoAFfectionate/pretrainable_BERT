"""Unified pre-training entry point for BERT and DeBERTa.

Select the architecture with ``--model {bert,deberta}``. The two share the
corpus format, tokenizer, NSP+MLM objectives, checkpoint layout and training
loop; only the encoder class and trainer wrapper differ.

    python pretrain.py --model bert    -c train.txt -v tokenizer.json ...
    python pretrain.py --model deberta -c train.txt -v tokenizer.json ...

DeBERTa-only flags (``--max_relative_positions``) are ignored when the
selected model is BERT.

Also exposes :func:`build_parser` / :func:`run` so ``main.py`` can dispatch
to this pipeline without re-parsing ``sys.argv`` twice.
"""

import argparse
import os
import sys
import warnings

from torch.utils.data import DataLoader

from dataset import BERTDataset, BPEVocab
from model import BERT, DeBERTa
from utils import (
    BERTTrainer, DeBERTaTrainer, TrainingLogger,
    apply_yaml_defaults, extract_config_path,
)

warnings.filterwarnings("ignore")


MODEL_REGISTRY = {
    "bert": {
        "encoder_cls": BERT,
        "trainer_cls": BERTTrainer,
        "task_name": "bert_pretraining",
        "log_prefix": "bert",
    },
    "deberta": {
        "encoder_cls": DeBERTa,
        "trainer_cls": DeBERTaTrainer,
        "task_name": "deberta_pretraining",
        "log_prefix": "deberta",
    },
}


def build_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """Return (or extend) an argparse parser for the pretraining command."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Unified BERT / DeBERTa pre-training")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (see config/*.yaml). "
                             "Values in the YAML override argparse defaults; "
                             "CLI flags still override the YAML.")

    parser.add_argument("-m", "--model", choices=list(MODEL_REGISTRY.keys()),
                        default="bert",
                        help="Which architecture to pre-train (default: bert)")

    parser.add_argument("-c", "--train_dataset", required=True, type=str,
                        help="Path to training dataset (sentence1 TAB sentence2 per line)")
    parser.add_argument("-t", "--test_dataset", type=str, default=None,
                        help="Path to test dataset (optional)")
    parser.add_argument("-v", "--vocab_path", required=True, type=str,
                        help="Path to tokenizer.json (trained by dataset.build_tokenizer)")

    parser.add_argument("-hs", "--hidden", type=int, default=768, help="Hidden size")
    parser.add_argument("-l", "--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_relative_positions", type=int, default=512,
                        help="Max relative positions (DeBERTa only; ignored for BERT)")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", "-gas", type=int, default=1,
                        help="Accumulate gradients over N micro-batches before each "
                             "optimizer step. Effective batch = batch_size * N. "
                             "Used to reach the paper's 256 effective batch on a "
                             "single GPU.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")

    parser.add_argument("--result_root", type=str, default="result",
                        help="Root directory for run outputs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of this run (default: {model}_<timestamp>)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loader workers")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Logging frequency in steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--cuda_devices", type=str, default=None,
                        help="Comma-separated list of CUDA device IDs")
    return parser


def _build_encoder(model_name: str, vocab_size: int, args):
    if model_name == "bert":
        return BERT(vocab_size=vocab_size, hidden=args.hidden, n_layers=args.layers,
                    attn_heads=args.attn_heads, dropout=args.dropout)
    if model_name == "deberta":
        return DeBERTa(vocab_size=vocab_size, hidden=args.hidden, n_layers=args.layers,
                       attn_heads=args.attn_heads, dropout=args.dropout,
                       max_relative_positions=args.max_relative_positions)
    raise ValueError(f"Unknown model: {model_name}")


def _model_args_for_config(model_name: str, vocab_size: int, args):
    cfg = {
        "vocab_size": vocab_size,
        "hidden": args.hidden,
        "n_layers": args.layers,
        "attn_heads": args.attn_heads,
        "dropout": args.dropout,
    }
    if model_name == "deberta":
        cfg["max_relative_positions"] = args.max_relative_positions
    return cfg


def run(args) -> None:
    """Execute the pretraining pipeline described by ``args``."""
    spec = MODEL_REGISTRY[args.model]

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    tlogger = TrainingLogger(result_root=args.result_root, run_name=args.run_name,
                             prefix=spec["log_prefix"])
    tlogger.info("Starting %s pre-training", args.model.upper())
    if getattr(args, "config", None):
        tlogger.info("Loaded defaults from config: %s", args.config)

    tlogger.info("Loading vocabulary from %s", args.vocab_path)
    vocab = BPEVocab.load_vocab(args.vocab_path)
    tlogger.info("Vocabulary size: %d", len(vocab))

    tlogger.info("Creating training dataset from %s", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len)

    test_dataset = None
    if args.test_dataset:
        tlogger.info("Creating test dataset from %s", args.test_dataset)
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    encoder = _build_encoder(args.model, len(vocab), args)

    # total_steps drives the scheduler and must count *optimizer* steps, which
    # fire once per accumulation window (plus one flush per epoch tail).
    accum = max(1, int(args.gradient_accumulation_steps))
    batches_per_epoch = len(train_loader)
    steps_per_epoch = (batches_per_epoch + accum - 1) // accum  # ceil for tail flush
    total_steps = max(1, args.epochs * steps_per_epoch)
    effective_bs = args.batch_size * accum
    tlogger.info(
        "Batch: micro=%d × accum=%d → effective=%d | %d batches/epoch → %d steps/epoch",
        args.batch_size, accum, effective_bs, batches_per_epoch, steps_per_epoch,
    )

    tlogger.save_config({
        "task": spec["task_name"],
        "model": args.model,
        "training_args": vars(args),
        "model_args": _model_args_for_config(args.model, len(vocab), args),
        "derived": {
            "train_batches_per_epoch": batches_per_epoch,
            "gradient_accumulation_steps": accum,
            "effective_batch_size": effective_bs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        },
    })

    trainer = spec["trainer_cls"](
        encoder, len(vocab),
        train_dataloader=train_loader, test_dataloader=test_loader,
        lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, total_steps=total_steps,
        gradient_accumulation_steps=accum,
        grad_clip=args.grad_clip, fp16=args.fp16,
        cuda_devices=args.cuda_devices, log_freq=args.log_freq,
        training_logger=tlogger,
    )

    start_epoch = 0
    if args.resume:
        last_epoch = trainer.load_checkpoint(args.resume)
        start_epoch = last_epoch + 1

    tlogger.info("Start Training...")
    for epoch in range(start_epoch, args.epochs):
        loss, _, _ = trainer.train(epoch)
        trainer.save_best(epoch, loss)
        if test_loader is not None:
            trainer.test(epoch)

    tlogger.info("Finish Training, best loss: %.3f", trainer.best_loss)
    tlogger.info("Best checkpoint: %s", tlogger.checkpoint_path(trainer.best_ckpt_name))
    tlogger.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # Two-phase: if a --config path is present, load it and overlay its values
    # onto the parser's defaults before the full parse. CLI flags still win.
    config_path = extract_config_path(argv)
    if config_path:
        apply_yaml_defaults(parser, config_path)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
