#!/usr/bin/env bash
# Install the optional VLA demo dependencies and cache its default assets.
#
# Usage:
#   ./vla/setup.sh
#   ./vla/setup.sh --model-only
#   MODEL_ID=org/my-finetune DATASET_ID=org/my-dataset ./vla/setup.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-lerobot/smolvla_libero}"
DATASET_ID="${DATASET_ID:-lerobot/libero}"
DOWNLOAD_DATASET=true

case "${1:-}" in
  "")
    ;;
  --model-only)
    DOWNLOAD_DATASET=false
    ;;
  -h|--help)
    sed -n '2,8p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    exit 2
    ;;
esac

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "lerobot[smolvla,dataset,libero]"

# LIBERO's first import asks an interactive question before creating this file.
# Create its default configuration during setup so the rollout script is non-interactive.
if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
  printf 'n\n' | "$VENV_DIR/bin/python" -c "import libero.libero"
fi

# Store downloads in the standard Hugging Face cache (or HF_HOME if configured).
"$VENV_DIR/bin/hf" download "$MODEL_ID" --repo-type model

if "$DOWNLOAD_DATASET"; then
  echo "Downloading $DATASET_ID. This can require substantial disk space."
  "$VENV_DIR/bin/hf" download "$DATASET_ID" --repo-type dataset
fi

cat <<EOF

Setup complete. Run:
  # Inspect one recorded dataset sample and its predicted action chunk.
  $VENV_DIR/bin/python $ROOT_DIR/vla/vla_demo_onesample.py

For a fine-tuned checkpoint or another dataset:
  $VENV_DIR/bin/python $ROOT_DIR/vla/vla_demo_onesample.py --model MODEL_ID --dataset DATASET_ID
EOF
