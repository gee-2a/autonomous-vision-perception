"""Shared configuration for the GTSRB traffic sign classifier."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_PATH = OUTPUT_DIR / "traffic_sign_classifier.keras"

DATASET_NAME = "gtsrb"
NUM_CLASSES = 43
IMAGE_SIZE = (224, 224)  # MobileNetV2's standard input resolution.
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
RANDOM_SEED = 42
