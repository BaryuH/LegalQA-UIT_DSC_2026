#!/usr/bin/env bash
# SS-04A — Bootstrap isolated Linux RTX4090 SEDAR-SFT environment.
# Run on Ubuntu 22.04 GPU server only. Does not modify system Python.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: SS-04A canonical bootstrap requires Linux (found $(uname -s))." >&2
  exit 1
fi

SEDAR_WORK_ROOT="${SEDAR_WORK_ROOT:-/mnt/F/sedar-legalqa}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${SEDAR_VENV_DIR:-$SEDAR_WORK_ROOT/venvs/sedar-sft}"

if [[ ! -d "$(dirname "$SEDAR_WORK_ROOT")" ]] && [[ ! -d "$SEDAR_WORK_ROOT" ]]; then
  echo "ERROR: SEDAR_WORK_ROOT parent missing: $SEDAR_WORK_ROOT" >&2
  echo "Set SEDAR_WORK_ROOT to an approved NVMe path (prefer /mnt/F/...)." >&2
  exit 1
fi

mkdir -p \
  "$SEDAR_WORK_ROOT/hf-cache" \
  "$SEDAR_WORK_ROOT/datasets-cache" \
  "$SEDAR_WORK_ROOT/torch-cache" \
  "$SEDAR_WORK_ROOT/processed" \
  "$SEDAR_WORK_ROOT/checkpoints" \
  "$SEDAR_WORK_ROOT/outputs" \
  "$(dirname "$VENV_DIR")"

export HF_HOME="$SEDAR_WORK_ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DATASETS_CACHE="$SEDAR_WORK_ROOT/datasets-cache"
export TORCH_HOME="$SEDAR_WORK_ROOT/torch-cache"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Install Python >=3.11 without touching system site-packages." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >=3.11 required, found {sys.version}")
print(f"OK Python {sys.version.split()[0]}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio
python -m pip install -r "$REPO_ROOT/requirements/sedar_sft_linux_cu124.txt"
python -m pip install -e "$REPO_ROOT.[sedar-sft,dev]"

python "$REPO_ROOT/scripts/sedar_sft/probe_environment.py" \
  --out "$REPO_ROOT/artifacts/sedar_sft/hardware/environment_probe.json"

echo "SS-04A bootstrap finished."
echo "Venv: $VENV_DIR"
echo "SEDAR_WORK_ROOT=$SEDAR_WORK_ROOT"
echo "Activate: source $VENV_DIR/bin/activate"
echo "Re-validate CUDA with the playbook torch snippet before marking Exit Gate PASS."
