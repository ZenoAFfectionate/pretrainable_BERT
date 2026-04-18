#!/usr/bin/env bash
# scripts/pretrain.sh — pre-train BERT or DeBERTa using a YAML profile.
#
# Usage:
#     bash scripts/pretrain.sh <bert|deberta> [mini|base|pro] [extra main.py args...]
#
# Examples:
#     bash scripts/pretrain.sh bert base                     # BERT-Base (paper config)
#     bash scripts/pretrain.sh deberta pro                   # DeBERTa-Large
#     bash scripts/pretrain.sh bert mini --epochs 3          # BERT-Mini, override epochs
#     CUDA_VISIBLE_DEVICES=1 bash scripts/pretrain.sh bert base --run_name my_exp
#
# Environment overrides:
#     CORPUS_TRAIN   — path to training corpus (default dataset/data/corpus/train.txt)
#     CORPUS_TEST    — path to test corpus     (default dataset/data/corpus/test.txt)
#     VOCAB_PATH     — path to tokenizer.json  (default dataset/data/tokenizer/tokenizer.json)
#     RUN_NAME       — run name                (default <model>_<tier>_<timestamp>)

set -euo pipefail

# Resolve repo root so the script works regardless of cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

usage() {
    cat <<EOF
Usage: bash scripts/pretrain.sh <bert|deberta> [mini|base|pro] [extra args...]

First positional arg:
    bert       Pre-train the BERT encoder (standard self-attention)
    deberta    Pre-train the DeBERTa encoder (disentangled attention + EMD)

Second positional arg (optional, default: base):
    mini       Fast smoke-test profile (~5M params, seq_len=128)
    base       Paper-aligned BERT-Base  / DeBERTa-Base (effective batch 256)
    pro        Paper-aligned BERT-Large / DeBERTa-Large (effective batch 256)

Any further args are forwarded verbatim to 'python main.py pretrain ...'.
EOF
}

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODEL="$1"; shift
TIER="${1:-base}"
# Only shift the tier off if it actually matches one of the known tiers;
# otherwise the user passed through-args directly after the model name.
case "${TIER}" in
    mini|base|pro) shift ;;
    *) TIER="base" ;;
esac

case "${MODEL}" in
    bert|deberta) ;;
    *)
        echo "error: first arg must be 'bert' or 'deberta', got '${MODEL}'" >&2
        usage
        exit 2
        ;;
esac

CONFIG="config/${MODEL}_${TIER}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "error: config file not found: ${CONFIG}" >&2
    exit 2
fi

# Resolve data paths (env-overridable).
CORPUS_TRAIN="${CORPUS_TRAIN:-dataset/data/corpus/train.txt}"
CORPUS_TEST="${CORPUS_TEST:-dataset/data/corpus/test.txt}"
VOCAB_PATH="${VOCAB_PATH:-dataset/data/tokenizer/tokenizer.json}"

for required in "${CORPUS_TRAIN}" "${VOCAB_PATH}"; do
    if [[ ! -f "${required}" ]]; then
        echo "error: required file missing: ${required}" >&2
        echo "       build it first (see README Step 1 / Step 2)" >&2
        exit 2
    fi
done

DEFAULT_RUN_NAME="${MODEL}_${TIER}_$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

echo "========================================================================"
echo " Pre-training: model=${MODEL}  tier=${TIER}  config=${CONFIG}"
echo " Corpus : ${CORPUS_TRAIN} (+ test: ${CORPUS_TEST})"
echo " Vocab  : ${VOCAB_PATH}"
echo " Run    : ${RUN_NAME}"
echo " Extra  : $*"
echo "========================================================================"

CMD=(python main.py pretrain
    --config "${CONFIG}"
    -c "${CORPUS_TRAIN}"
    -v "${VOCAB_PATH}"
    --run_name "${RUN_NAME}")

# Include test set only if it exists.
if [[ -f "${CORPUS_TEST}" ]]; then
    CMD+=(-t "${CORPUS_TEST}")
fi

CMD+=("$@")

exec "${CMD[@]}"
