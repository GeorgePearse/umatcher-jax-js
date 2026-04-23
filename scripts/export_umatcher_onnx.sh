#!/usr/bin/env bash
# Export UMatcher ONNX weights from the upstream PyTorch repo.
#
# Usage:
#   ./scripts/export_umatcher_onnx.sh [/path/to/checkpoint.pth]
#
# If a checkpoint path is omitted, the script uses data/best.pth (the default
# used by the upstream export_onnx.py). The ONNX files end up under
# public/models/ where the frontend looks for them.

set -euo pipefail

CKPT="${1:-data/best.pth}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/.umatcher_src"
OUT_DIR="${REPO_ROOT}/public/models"

mkdir -p "${OUT_DIR}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Cloning upstream UMatcher repo into ${SRC_DIR}..."
  git clone --depth 1 https://github.com/aemior/UMatcher "${SRC_DIR}"
fi

pushd "${SRC_DIR}" > /dev/null
python scripts/export_onnx.py --ckpt "${CKPT}" --export_dir "${OUT_DIR}"
popd > /dev/null

echo "ONNX weights written to ${OUT_DIR}:"
ls -lh "${OUT_DIR}"/*.onnx || true
