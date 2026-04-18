"""Fine-tune a pre-trained BERT or DeBERTa encoder on SQuAD v1.1 QA.

Mirrors the structure of ``pretrain.py``: a ``build_parser()`` / ``run(args)``
pair that ``main.py`` dispatches into, plus a ``__main__`` entry-point for
standalone use.

The pretrained architecture (BERT vs DeBERTa) is auto-detected from the
``config`` block in the pretraining checkpoint; pass ``--model`` to override
the detection.

A QA head (``Linear(hidden -> 2)``) is attached on top of the loaded encoder,
the whole model is trained end-to-end with AdamW + warmup-then-linear decay,
and the best-F1 checkpoint (``best_model.pt``) plus a final checkpoint
(``last_model.pt``) are written under ``result/<run_name>/checkpoints/``.
The saved state is a complete :class:`QAModel` state dict plus a ``config``
block; ``evaluate.py`` rebuilds the exact same model from it.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import tqdm
from torch import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    BPEVocab, SquadExample, SquadFeature, SquadFeatureDataset,
    convert_examples_to_features, load_squad_examples, squad_metrics,
)
from model import QAModel, load_pretrained_encoder
from utils import (
    TrainingLogger, get_warmup_linear_schedule, parse_cuda_devices,
    apply_yaml_defaults, extract_config_path,
)

warnings.filterwarnings("ignore")


def build_parser(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Fine-tune BERT/DeBERTa on SQuAD v1.1")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (see config/finetune_squad.yaml). "
                             "YAML values override argparse defaults; CLI flags override YAML.")

    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to pretraining checkpoint (best_model.pt)")
    parser.add_argument("--vocab_path", required=True, type=str,
                        help="Path to tokenizer.json")
    parser.add_argument("--train_file", required=True, type=str,
                        help="Path to SQuAD train-v1.1.json")
    parser.add_argument("--dev_file", type=str, default=None,
                        help="Path to SQuAD dev-v1.1.json (optional; enables per-epoch eval)")

    parser.add_argument("-m", "--model", choices=["bert", "deberta"], default=None,
                        help="Override architecture detection (usually not needed)")

    parser.add_argument("--seq_len", type=int, default=384, help="Max sequence length")
    parser.add_argument("--max_query_len", type=int, default=64,
                        help="Truncate question tokens to this length")
    parser.add_argument("--max_answer_tokens", type=int, default=30,
                        help="Maximum predicted answer length in sub-words (eval only)")

    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", "-gas", type=int, default=1,
                        help="Accumulate gradients over N micro-batches per optimizer step. "
                             "Effective batch = batch_size * N.")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cuda_devices", type=str, default=None)

    parser.add_argument("--result_root", type=str, default="result")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of this run (default: finetune_<timestamp>)")

    parser.add_argument("--max_train_examples", type=int, default=-1,
                        help="Cap on training examples (-1 = all; useful for smoke tests)")
    parser.add_argument("--max_dev_examples", type=int, default=-1,
                        help="Cap on dev examples (-1 = all)")
    return parser


# ---------------------------------------------------------------------------
# Training / eval helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                     loss_fn, args, epoch, tlogger, step_counter) -> float:
    model.train()
    running_loss = 0.0
    it = tqdm.tqdm(enumerate(loader), total=len(loader),
                   desc=f"finetune epoch {epoch:02d}",
                   bar_format="{l_bar}{bar:30}{r_bar}", dynamic_ncols=True)

    accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    last_idx = len(loader) - 1
    # Flush any stale grads from a previous epoch before opening a window.
    optimizer.zero_grad(set_to_none=True)

    for i, batch in it:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        start_pos = batch["start_position"].to(device, non_blocking=True)
        end_pos = batch["end_position"].to(device, non_blocking=True)

        with amp.autocast("cuda", enabled=args.fp16):
            start_logits, end_logits = model(input_ids, segment_ids)
            loss = 0.5 * (loss_fn(start_logits, start_pos)
                          + loss_fn(end_logits, end_pos))

        scaler.scale(loss / accum).backward()

        # Step only on accumulation boundaries, plus a tail flush at epoch end.
        is_boundary = ((i + 1) % accum == 0) or (i == last_idx)
        if is_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_counter[0] += 1

        running_loss += loss.item()

        if i % args.log_freq == 0 or i == last_idx:
            avg = running_loss / (i + 1)
            lr = scheduler.get_last_lr()[0]
            it.set_postfix({"LR": f"{lr:.2e}", "Loss": f"{avg:.4f}"})
            tlogger.log_step(epoch=epoch, step=step_counter[0],
                             lr=lr, batch_loss=loss.item(),
                             running_loss=avg)
    return running_loss / max(1, len(loader))


@torch.no_grad()
def run_qa_evaluation(model, features: List[SquadFeature],
                      examples: List[SquadExample], loader, device,
                      fp16: bool, max_answer_tokens: int):
    """Predict + score on the dev set. Returns (metrics, predictions)."""
    model.eval()
    predictions: Dict[str, str] = {}

    for batch in tqdm.tqdm(loader, desc="evaluating",
                           bar_format="{l_bar}{bar:30}{r_bar}",
                           dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        feat_idx = batch["feature_index"].tolist()

        with amp.autocast("cuda", enabled=fp16):
            start_logits, end_logits = model(input_ids, segment_ids)
        start_logits = start_logits.float().cpu()
        end_logits = end_logits.float().cpu()

        for bi, fi in enumerate(feat_idx):
            feat = features[fi]
            ex = examples[feat.example_index]
            s_logits = start_logits[bi]
            e_logits = end_logits[bi]

            c_start = feat.context_start_in_input
            c_len = len(feat.context_token_offsets)
            s_scores = s_logits[c_start:c_start + c_len]
            e_scores = e_logits[c_start:c_start + c_len]

            k = min(20, c_len)
            top_s = torch.topk(s_scores, k).indices.tolist()
            top_e = torch.topk(e_scores, k).indices.tolist()

            best_score = -1e30
            best_span = None
            for si in top_s:
                for ei in top_e:
                    if ei < si or ei - si + 1 > max_answer_tokens:
                        continue
                    score = s_scores[si].item() + e_scores[ei].item()
                    if score > best_score:
                        best_score = score
                        best_span = (si, ei)

            if best_span is None:
                predictions[ex.qid] = ""
                continue
            si, ei = best_span
            char_start = feat.context_token_offsets[si][0]
            char_end = feat.context_token_offsets[ei][1]
            predictions[ex.qid] = ex.context[char_start:char_end].strip()

    metrics = squad_metrics(predictions, examples)
    return metrics, predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    tlogger = TrainingLogger(result_root=args.result_root, run_name=args.run_name,
                             prefix="finetune")
    tlogger.info("Starting SQuAD v1.1 fine-tuning")
    if getattr(args, "config", None):
        tlogger.info("Loaded defaults from config: %s", args.config)

    tlogger.info("Loading vocabulary from %s", args.vocab_path)
    vocab = BPEVocab.load_vocab(args.vocab_path)
    tlogger.info("Vocabulary size: %d", len(vocab))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlogger.info("Using device: %s", device)

    tlogger.info("Loading pretraining checkpoint %s", args.checkpoint)
    pre_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder, model_type = load_pretrained_encoder(
        pre_ckpt, vocab_size=len(vocab),
        override_model_type=args.model, logger=tlogger,
    )
    model = QAModel(encoder).to(device)

    dp_enabled = parse_cuda_devices(args.cuda_devices) and torch.cuda.device_count() > 1
    if dp_enabled:
        tlogger.info("Wrapping model in DataParallel over %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    # -------------------------------------------------------------- features
    tlogger.info("Loading SQuAD train from %s", args.train_file)
    train_examples = load_squad_examples(args.train_file)
    if args.max_train_examples > 0:
        train_examples = train_examples[:args.max_train_examples]
    tlogger.info("Loaded %d train examples", len(train_examples))

    train_features = convert_examples_to_features(
        train_examples, vocab, seq_len=args.seq_len, is_training=True,
        max_query_len=args.max_query_len, logger=tlogger,
    )
    if not train_features:
        raise RuntimeError("No trainable SQuAD features — corpus too small or seq_len too short")

    train_loader = DataLoader(
        SquadFeatureDataset(train_features, is_training=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    dev_examples: List[SquadExample] = []
    dev_features: List[SquadFeature] = []
    dev_loader = None
    if args.dev_file:
        tlogger.info("Loading SQuAD dev from %s", args.dev_file)
        dev_examples = load_squad_examples(args.dev_file)
        if args.max_dev_examples > 0:
            dev_examples = dev_examples[:args.max_dev_examples]
        tlogger.info("Loaded %d dev examples", len(dev_examples))
        dev_features = convert_examples_to_features(
            dev_examples, vocab, seq_len=args.seq_len, is_training=False,
            max_query_len=args.max_query_len, logger=tlogger,
        )
        dev_loader = DataLoader(
            SquadFeatureDataset(dev_features, is_training=False),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    # Encoder config used later for rebuild in evaluate.py.
    encoder_cfg = {
        "hidden": encoder.hidden,
        "layers": encoder.n_layers,
        "attn_heads": encoder.attn_heads,
        "dropout": 0.1,
    }
    if model_type == "deberta":
        encoder_cfg["max_relative_positions"] = getattr(encoder, "max_relative_positions", 512)

    tlogger.save_config({
        "task": "squad_v1.1_finetune",
        "model_type": model_type,
        "pretrain_checkpoint": str(args.checkpoint),
        "pretrain_config": pre_ckpt.get("config", {}),
        "encoder_config": encoder_cfg,
        "training_args": vars(args),
        "train_features": len(train_features),
        "dev_features": len(dev_features),
    })

    # -------------------------------------------------------------- optimizer
    accum = max(1, int(args.gradient_accumulation_steps))
    batches_per_epoch = max(1, len(train_loader))
    steps_per_epoch = (batches_per_epoch + accum - 1) // accum
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)
    effective_bs = args.batch_size * accum
    tlogger.info(
        "Batch: micro=%d × accum=%d → effective=%d | %d batches/epoch → %d steps/epoch",
        args.batch_size, accum, effective_bs, batches_per_epoch, steps_per_epoch,
    )

    no_decay = ("bias", "LayerNorm.weight")
    named = list(model.named_parameters())
    grouped = [
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        grouped, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8,
        fused=torch.cuda.is_available(),
    )
    scheduler = get_warmup_linear_schedule(optimizer, warmup_steps=warmup_steps,
                                           total_steps=total_steps)
    scaler = amp.GradScaler("cuda", enabled=args.fp16)
    loss_fn = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    tlogger.info("Total parameters: %.2fM", n_params / 1e6)
    tlogger.info("Fine-tuning for %d epochs (%d total steps, %d warmup)",
                 args.epochs, total_steps, warmup_steps)

    best_f1 = -1.0
    step_counter = [0]

    for epoch in range(args.epochs):
        avg_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            loss_fn, args, epoch, tlogger, step_counter,
        )
        tlogger.log_epoch(epoch=epoch, split="train", avg_loss=avg_loss)

        metrics = None
        if dev_loader is not None:
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            metrics, _ = run_qa_evaluation(
                eval_model, dev_features, dev_examples, dev_loader, device,
                fp16=args.fp16, max_answer_tokens=args.max_answer_tokens,
            )
            tlogger.log_epoch(epoch=epoch, split="dev",
                              exact_match=metrics["exact_match"],
                              f1=metrics["f1"], count=metrics["count"])
            tlogger.info("Epoch %d dev: EM=%.2f F1=%.2f",
                         epoch, metrics["exact_match"], metrics["f1"])

        # Checkpointing: best-F1 on dev (or latest-loss if no dev).
        save_payload_model = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint = {
            "epoch": epoch,
            "step": step_counter[0],
            "model_state": save_payload_model.state_dict(),
            "config": {
                "model_type": model_type,
                **encoder_cfg,
                "seq_len": args.seq_len,
                "max_query_len": args.max_query_len,
                "max_answer_tokens": args.max_answer_tokens,
            },
            "metrics": metrics,
        }
        if metrics is not None and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_path = tlogger.checkpoint_path("best_model.pt")
            torch.save(checkpoint, best_path)
            tlogger.info("New best dev F1 %.2f — saved %s", best_f1, best_path)
        elif metrics is None:
            # No dev set: just save the latest and treat it as best.
            best_path = tlogger.checkpoint_path("best_model.pt")
            torch.save(checkpoint, best_path)

        last_path = tlogger.checkpoint_path("last_model.pt")
        torch.save(checkpoint, last_path)

    tlogger.info("Finish fine-tuning. Best dev F1: %.2f", best_f1 if best_f1 >= 0 else float("nan"))
    tlogger.info("Best checkpoint: %s", tlogger.checkpoint_path("best_model.pt"))
    tlogger.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    config_path = extract_config_path(argv)
    if config_path:
        apply_yaml_defaults(parser, config_path)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
