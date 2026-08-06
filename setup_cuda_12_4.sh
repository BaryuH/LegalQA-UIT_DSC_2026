#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dsc"

echo "=================================================="
echo "🔥 [1/3] Setting Up Compilers & CUDA Toolkit (nvcc, gcc, g++)"
echo "=================================================="

# Check if apt-get is available to install system compilers
if command -v apt-get &> /dev/null; then
    echo "📌 Installing/Verifying gcc, g++, build-essential, and nvidia-cuda-toolkit via apt..."
    export DEBIAN_FRONTEND=noninteractive
    if command -v sudo &> /dev/null; then
        sudo apt-get update -y
        sudo apt-get install -y --no-install-recommends \
            gcc g++ build-essential nvidia-cuda-toolkit || true
    else
        apt-get update -y
        apt-get install -y --no-install-recommends \
            gcc g++ build-essential nvidia-cuda-toolkit || true
    fi
fi

# Ensure conda env is active if conda exists
if command -v conda &> /dev/null; then
    CONDA_DIR="$(dirname "$(dirname "$(command -v conda)")")"
    if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        source "$CONDA_DIR/etc/profile.d/conda.sh"
    fi
    set +u
    conda activate "$ENV_NAME" 2>/dev/null || true
    set -u
fi

echo ""
echo "=================================================="
echo "⚡ [2/3] Installing PyTorch for CUDA 12.4"
echo "=================================================="

echo "📌 Installing PyTorch, torchvision, torchaudio for CUDA 12.4..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=================================================="
echo "🔍 [3/3] Comprehensive Environment Verification"
echo "=================================================="

echo "--- 1. GCC Compiler Check ---"
if command -v gcc &> /dev/null; then
    gcc --version | head -n 1
else
    echo "⚠️ Warning: 'gcc' not found in PATH."
fi

echo ""
echo "--- 2. G++ Compiler Check ---"
if command -v g++ &> /dev/null; then
    g++ --version | head -n 1
else
    echo "⚠️ Warning: 'g++' not found in PATH."
fi

echo ""
echo "--- 3. NVCC CUDA Compiler Check ---"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep -i "release" || nvcc --version
else
    echo "⚠️ Note: 'nvcc' binary not in PATH (Checking system CUDA paths /usr/local/cuda/bin/nvcc)..."
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        /usr/local/cuda/bin/nvcc --version | grep -i "release"
    fi
fi

echo ""
echo "--- 4. NVIDIA Driver (nvidia-smi) Check ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠️ Note: 'nvidia-smi' not found or NVIDIA driver not loaded."
fi

echo ""
echo "--- 5. PyTorch CUDA Integration Check ---"
python -c '
import sys
import torch

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Built with CUDA: {torch.version.cuda}")
cuda_ok = torch.cuda.is_available()
print(f"CUDA Available to PyTorch: {cuda_ok}")

if cuda_ok:
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device [{i}]: {torch.cuda.get_device_name(i)}")
    # Smoke test tensor creation on GPU
    try:
        x = torch.ones((2, 2), device="cuda")
        print("✅ GPU Tensor Allocation Smoke Test PASSED!")
    except Exception as e:
        print(f"❌ GPU Tensor Allocation Smoke Test FAILED: {e}")
else:
    print("⚠️ WARNING: PyTorch cannot access GPU. Please check NVIDIA driver installation.")
'

echo ""
echo "=================================================="
echo "🎉 CUDA 12.4 Setup & Verification Finished!"
echo "=================================================="
