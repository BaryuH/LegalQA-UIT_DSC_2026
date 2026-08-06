#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dsc"
PYTHON_VERSION="3.10"

echo "=================================================="
echo "🚀 [1/4] Checking System & OS Environment"
echo "=================================================="

IS_UBUNTU=false
if [ -f /etc/os-release ]; then
    if grep -qi "ubuntu" /etc/os-release; then
        IS_UBUNTU=true
    fi
fi

if [ "$IS_UBUNTU" = true ]; then
    echo "📌 Detected Ubuntu OS. Installing system packages (rclone, wget, curl, git, build-essential)..."
    export DEBIAN_FRONTEND=noninteractive
    if command -v sudo &> /dev/null; then
        sudo apt-get update -y
        sudo apt-get install -y --no-install-recommends \
            wget curl git rclone build-essential ca-certificates unzip
    else
        apt-get update -y
        apt-get install -y --no-install-recommends \
            wget curl git rclone build-essential ca-certificates unzip
    fi
    echo "✅ Ubuntu system dependencies & rclone installed successfully."
else
    echo "📌 Non-Ubuntu system detected. Proceeding with Miniconda installation & conda env setup..."
fi

echo ""
echo "=================================================="
echo "🐍 [2/4] Checking & Installing Miniconda"
echo "=================================================="

# Function to locate conda executable
find_conda() {
    if command -v conda &> /dev/null; then
        echo "$(command -v conda)"
    elif [ -f "$HOME/miniconda3/bin/conda" ]; then
        echo "$HOME/miniconda3/bin/conda"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        echo "$HOME/anaconda3/bin/conda"
    elif [ -f "/opt/conda/bin/conda" ]; then
        echo "/opt/conda/bin/conda"
    else
        echo ""
    fi
}

CONDA_BIN="$(find_conda)"

if [ -z "$CONDA_BIN" ]; then
    echo "📌 Conda not found. Downloading & installing Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$MINICONDA_INSTALLER"
    
    CONDA_BIN="$HOME/miniconda3/bin/conda"
    echo "✅ Miniconda installed to $HOME/miniconda3"
else
    echo "✅ Found existing Conda executable at: $CONDA_BIN"
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA_BIN")")"
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
fi

echo ""
echo "=================================================="
echo "⚡ [3/4] Creating & Activating Conda Env '$ENV_NAME' (Python $PYTHON_VERSION)"
echo "=================================================="

if "$CONDA_BIN" env list | grep -qE "^${ENV_NAME}\s"; then
    echo "📌 Conda environment '$ENV_NAME' already exists."
else
    echo "📌 Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    "$CONDA_BIN" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "📌 Activating Conda environment '$ENV_NAME'..."
set +u
conda activate "$ENV_NAME" || source "$CONDA_DIR/bin/activate" "$ENV_NAME"
set -u

echo "✅ Active Python path: $(which python)"
echo "✅ Active Python version: $(python --version)"

echo ""
echo "=================================================="
echo "📦 [4/4] Installing Python Requirements & Package"
echo "=================================================="

pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    echo "📌 Installing packages from requirements.txt..."
    pip install -r requirements.txt
fi

echo "📌 Installing project in editable mode..."
pip install -e .

echo ""
echo "=================================================="
echo "🎉 SETUP Completed Successfully!"
echo "Environment '$ENV_NAME' is ready with Python $PYTHON_VERSION."
echo "To activate manually run: conda activate $ENV_NAME"
echo "=================================================="
