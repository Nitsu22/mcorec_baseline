#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="${ROOT_DIR}/data-bin/cache"
HF_ROOT="${CACHE_ROOT}/hf"

mkdir -p "${HF_ROOT}/datasets" "${HF_ROOT}/hub" "${HF_ROOT}/modules" "${CACHE_ROOT}/tmp"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  if [[ -f "${HF_ROOT}/token" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$(cat "${HF_ROOT}/token")"
  elif [[ -f "${HF_ROOT}/hub/token" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$(cat "${HF_ROOT}/hub/token")"
  else
    echo "Hugging Face token not found. Run:" >&2
    echo "  HF_HOME=${HF_ROOT} huggingface-cli login" >&2
    echo "or set HUGGINGFACE_HUB_TOKEN/HF_TOKEN in the environment." >&2
    exit 1
  fi
fi

export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"

HF_HOME="${HF_ROOT}" \
HF_DATASETS_CACHE="${HF_ROOT}/datasets" \
HF_HUB_CACHE="${HF_ROOT}/hub" \
HF_MODULES_CACHE="${HF_ROOT}/modules" \
TMPDIR="${CACHE_ROOT}/tmp" \
HF_TOKEN="${HF_TOKEN}" \
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN}" \
python "${ROOT_DIR}/script/precompute_muse_features.py" \
  --cache_dir "${CACHE_ROOT}" \
  --muse_cache_dir "${CACHE_ROOT}/muse_lip" \
  --include_mcorec \
  --muse_scope mcorec \
  --visual_frontend_ckpt "${ROOT_DIR}/../MuSE/pretrain_networks/visual_frontend.pt" \
  --device cuda
