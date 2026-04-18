#!/usr/bin/env bash
# scripts/finetune.sh — fine-tune a pre-trained encoder on SQuAD v1.1.
#
# Usage:
#     bash scripts/finetune.sh <checkpoint_path_or_run_name> [extra main.py args...]
#
# The first positional arg can be either:
#   (1) a direct path to a pretraining checkpoint, e.g.
#         result/bert_base_paper/checkpoints/best_model.pt
#   (2) a run name — the script resolves it to
#         result/<run_name>/checkpoints/best_model.pt
#
# The architecture (bert vs deberta) is auto-detected from the checkpoint —
# you do NOT pass it here. To force a particular architecture, append
# '--model bert' or '--model deberta' to the command.
#
# Examples:
#     bash scripts/finetune.sh bert_base_paper
#     bash scripts/finetune.sh result/deberta_base_paper/checkpoints/best_model.pt
#     bash scripts/finetune.sh bert_base_paper --epochs 3 --lr 5e-5
#     CUDA_VISIBLE_DEVICES=1 bash scripts/finetune.sh bert_base_paper
#
# Environment overrides:
#     VOCAB_PATH   — tokenizer.json (default dataset/data/tokenizer/tokenizer.json)
#     SQUAD_TRAIN  — SQuAD train JSON (default dataset/tune/squad/train-v1.1.json)
#     SQUAD_DEV    — SQuAD dev JSON   (default dataset/tune/squad/dev-v1.1.json)
#     RUN_NAME     — run name         (default squad_<source>_<timestamp>)
#     CONFIG       — YAML config      (default config/finetune_squad.yaml)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

usage() {
    cat <<EOF
Usage: bash scripts/finetune.sh <checkpoint_or_run_name> [extra args...]

First positional arg (required):
    Either a full checkpoint path (*.pt) or a run name under result/.
    The script accepts both forms and resolves the run name to
    result/<name>/checkpoints/best_model.pt.

The architecture (bert/deberta) is auto-detected from the checkpoint.

Extra args are forwarded to 'python main.py finetune ...'.
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CKPT_INPUT="$1"; shift

# Resolve to a checkpoint path.
if [[ "${CKPT_INPUT}" == *.pt ]] && [[ -f "${CKPT_INPUT}" ]]; then
    CHECKPOINT="${CKPT_INPUT}"
elif [[ -f "result/${CKPT_INPUT}/checkpoints/best_model.pt" ]]; then
    CHECKPOINT="result/${CKPT_INPUT}/checkpoints/best_model.pt"
else
    echo "error: cannot resolve checkpoint from '${CKPT_INPUT}'" >&2
    echo "  tried: ${CKPT_INPUT}" >&2
    echo "  tried: result/${CKPT_INPUT}/checkpoints/best_model.pt" >&2
    exit 2
fi

VOCAB_PATH="${VOCAB_PATH:-dataset/data/tokenizer/tokenizer.json}"
SQUAD_TRAIN="${SQUAD_TRAIN:-dataset/tune/squad/train-v1.1.json}"
SQUAD_DEV="${SQUAD_DEV:-dataset/tune/squad/dev-v1.1.json}"
CONFIG="${CONFIG:-config/finetune_squad.yaml}"

for required in "${VOCAB_PATH}" "${SQUAD_TRAIN}" "${SQUAD_DEV}" "${CONFIG}"; do
    if [[ ! -f "${required}" ]]; then
        echo "error: required file missing: ${required}" >&2
        echo "       (see README Step 2 for vocab, Step 4 for SQuAD)" >&2
        exit 2
    fi
done

# Derive a default run name from the pretraining run (if we were given one).
SOURCE_TAG="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
DEFAULT_RUN_NAME="squad_${SOURCE_TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

echo "========================================================================"
echo " Fine-tuning on SQuAD v1.1"
echo " Checkpoint: ${CHECKPOINT}"
echo " Config    : ${CONFIG}"
echo " Vocab     : ${VOCAB_PATH}"
echo " Train/Dev : ${SQUAD_TRAIN}"
echo "             ${SQUAD_DEV}"
echo " Run       : ${RUN_NAME}"
echo " Extra     : $*"
echo "========================================================================"

exec python main.py finetune \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --vocab_path "${VOCAB_PATH}" \
    --train_file "${SQUAD_TRAIN}" \
    --dev_file "${SQUAD_DEV}" \
    --run_name "${RUN_NAME}" \
    "$@"
