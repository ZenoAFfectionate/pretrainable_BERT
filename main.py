"""Unified CLI entry-point dispatching to pretrain / finetune / evaluate.

Each sub-command forwards its arguments to the matching module's
``run(args)`` function, so the flags you see under::

    python main.py pretrain --help
    python main.py finetune --help
    python main.py evaluate --help

are exactly the same as the standalone scripts. Architecture selection
(``--model bert`` vs ``--model deberta``) lives on the ``pretrain`` command;
``finetune`` and ``evaluate`` auto-detect the architecture from the
checkpoint they load.

Examples
--------
Pre-train DeBERTa::

    python main.py pretrain --model deberta \\
        -c dataset/data/corpus/train.txt \\
        -v dataset/data/tokenizer/tokenizer.json \\
        --epochs 3 --fp16

Fine-tune on SQuAD with the resulting checkpoint::

    python main.py finetune \\
        --checkpoint result/deberta_<ts>/checkpoints/best_model.pt \\
        --vocab_path dataset/data/tokenizer/tokenizer.json \\
        --train_file dataset/tune/squad/train-v1.1.json \\
        --dev_file   dataset/tune/squad/dev-v1.1.json \\
        --epochs 2 --fp16

Evaluate the fine-tuned model::

    python main.py evaluate \\
        --checkpoint result/finetune_<ts>/checkpoints/best_model.pt \\
        --vocab_path dataset/data/tokenizer/tokenizer.json \\
        --dev_file   dataset/tune/squad/dev-v1.1.json
"""

import argparse
import sys

import pretrain
import finetune
import evaluate

from utils import apply_yaml_defaults, extract_config_path


COMMANDS = {
    "pretrain": (pretrain.build_parser, pretrain.run,
                 "Pre-train a BERT or DeBERTa encoder (NSP + MLM)"),
    "finetune": (finetune.build_parser, finetune.run,
                 "Fine-tune a pre-trained encoder on SQuAD v1.1"),
    "evaluate": (evaluate.build_parser, evaluate.run,
                 "Evaluate a fine-tuned QA model on SQuAD v1.1"),
}


def build_parser(argv=None) -> argparse.ArgumentParser:
    """Build the top-level parser. If ``argv`` is given and names a command
    that registered ``--config``, the referenced YAML is overlaid as defaults
    onto that subparser before ``parse_args`` runs."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified entry-point for BERT / DeBERTa pre-training, "
                    "SQuAD fine-tuning, and SQuAD evaluation.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    active_cmd = argv[0] if argv else None

    for name, (build, _run, help_text) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_text, description=help_text)
        build(sp)
        # Apply YAML defaults to the active subparser so the final parse sees
        # them as baseline values (CLI flags further in argv still win).
        if name == active_cmd and argv is not None:
            cfg_path = extract_config_path(argv[1:])
            if cfg_path:
                apply_yaml_defaults(sp, cfg_path)

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().parse_args(argv)
        return

    parser = build_parser(argv)
    args = parser.parse_args(argv)
    _, run_fn, _ = COMMANDS[args.command]
    run_fn(args)


if __name__ == "__main__":
    main()
