"""Download and prepare GTSRB from Kaggle for TensorFlow training."""
import csv
import random
from pathlib import Path

import kagglehub
import tensorflow as tf

from config import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED, VALIDATION_SPLIT


def _find_training_csv(dataset_dir):
    """Locate the GTSRB training annotations in common Kaggle folder layouts."""
    candidates = list(Path(dataset_dir).rglob("Train.csv"))
    if not candidates:
        candidates = list(Path(dataset_dir).rglob("train.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find Train.csv under {dataset_dir}")
    return candidates[0]


def _read_examples(dataset_dir):
    """Read CSV image paths and class IDs (CSV paths are relative to its folder)."""
    annotation_file = _find_training_csv(dataset_dir)
    examples = []
    with annotation_file.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            relative_path = row.get("Path") or row.get("path")
            label = row.get("ClassId") or row.get("ClassID") or row.get("class_id")
            if relative_path is None or label is None:
                raise ValueError("GTSRB CSV must contain Path and ClassId columns")
            image_path = (annotation_file.parent / relative_path).resolve()
            if not image_path.exists():
                # Some Kaggle archives keep CSV paths relative to the dataset root.
                image_path = (Path(dataset_dir) / relative_path).resolve()
            examples.append((str(image_path), int(label)))
    if not examples:
        raise ValueError(f"No training examples found in {annotation_file}")
    return examples


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    # MobileNetV2 expects pixels scaled to [-1, 1].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image, tf.float32)
    )
    return image, tf.cast(label, tf.int32)


def load_datasets():
    """KaggleHub downloads GTSRB on first use; return shuffled train/validation sets."""
    dataset_dir = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    )
    print("Path to dataset files:", dataset_dir)

    examples = _read_examples(dataset_dir)
    random.Random(RANDOM_SEED).shuffle(examples)
    split_at = int(len(examples) * (1 - VALIDATION_SPLIT))
    train_examples, validation_examples = examples[:split_at], examples[split_at:]

    autotune = tf.data.AUTOTUNE

    def make_dataset(items, training):
        paths, labels = zip(*items)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
        if training:
            ds = ds.shuffle(len(items), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        return (
            ds.map(_load_image, num_parallel_calls=autotune)
            .batch(BATCH_SIZE)
            .prefetch(autotune)
        )

    return make_dataset(train_examples, True), make_dataset(validation_examples, False)
