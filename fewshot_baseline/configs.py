"""Configuration settings and Hugging Face model list for fewshot baseline."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "fewshot_baseline"

# Data file paths
WARMUP_PATH = DATA_DIR / "warmup.json"
PUBLIC_PATH = DATA_DIR / "public-official.json"
TRAIN_PATH = DATA_DIR / "train.json"

# Hugging Face Models list as requested
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Ministral-3b-instruct-2512",
    "Aimin12/Qwen3-4B-Thinking-2507-Distill-Claude-Opus-4.6-Reasoning-Abliterated",
]

# Generation & Hardware Settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.95
TRUST_REMOTE_CODE = True

# HF Token if needed from environment
HF_TOKEN = os.getenv("HF_TOKEN", None)
