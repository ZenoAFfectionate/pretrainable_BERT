#!/usr/bin/env bash
# scripts/evaluate.sh — evaluate a SQuAD-fine-tuned QA model on the dev split.
#
# Usage:
#     bash scripts/evaluate.sh <checkpoint_path_or_run_name> [extra main.py args...]
#
# The first positional arg can be either:
#   (1) a direct path to a finetune checkpoint, e.g.
#         result/squad_bert_base/checkpoints/best_model.pt
#   (2) a run name — resolved to
#         result/<run_name>/checkpoints/best_model.pt
#
# The architecture (bert vs deberta) is auto-detected from the checkpoint.
#
# Examples:
#     bash scripts/evaluate.sh squad_bert_base
#     bash scripts/evaluate.sh result/squad_deberta_base/checkpoints/best_model.pt
#     bash scripts/evaluate.sh squad_bert_base --max_dev_examples 500
#     CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate.sh squad_bert_base
#
# Environment overrides:
#     VOCAB_PATH — tokenizer.json (default dataset/data/tokenizer/tokenizer.json)
#     SQUAD_DEV  — SQuAD dev JSON (default dataset/tune/squad/dev-v1.1.json)
#     RUN_NAME   — run name       (default eval_<source>_<timestamp>)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

usage() {
    cat <<EOF
Usage: bash scripts/evaluate.sh <checkpoint_or_run_name> [extra args...]

First positional arg (required):
    Either a full finetune checkpoint path (*.pt) or a run name under result/.

Extra args are forwarded to 'python main.py evaluate ...'.
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CKPT_INPUT="$1"; shift

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
SQUAD_DEV="${SQUAD_DEV:-dataset/tune/squad/dev-v1.1.json}"

for required in "${VOCAB_PATH}" "${SQUAD_DEV}"; do
    if [[ ! -f "${required}" ]]; then
        echo "error: required file missing: ${required}" >&2
        exit 2
    fi
done

SOURCE_TAG="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
DEFAULT_RUN_NAME="eval_${SOURCE_TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

echo "========================================================================"
echo " Evaluating QA model on SQuAD v1.1 dev"
echo " Checkpoint: ${CHECKPOINT}"
echo " Vocab     : ${VOCAB_PATH}"
echo " Dev       : ${SQUAD_DEV}"
echo " Run       : ${RUN_NAME}"
echo " Extra     : $*"
echo "========================================================================"

exec python main.py evaluate \
    --checkpoint "${CHECKPOINT}" \
    --vocab_path "${VOCAB_PATH}" \
    --dev_file "${SQUAD_DEV}" \
    --run_name "${RUN_NAME}" \
    --fp16 \
    "$@"
