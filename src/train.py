"""Train the traffic sign classifier in frozen-head and fine-tuning phases."""
import tensorflow as tf

from config import (
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    LOG_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
)
from data_loader import load_datasets
from model import build_model


def _compile(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, validation_ds = load_datasets()
    model, base_model = build_model()

    # TensorBoard records metrics for both phases in the same run directory.
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=str(LOG_DIR), histogram_freq=0, update_freq="epoch"
    )

    # Phase 1: keep pretrained convolutional features fixed while learning the head.
    _compile(model, HEAD_LEARNING_RATE)
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=HEAD_EPOCHS,
        callbacks=[tensorboard],
    )

    # Phase 2: fine-tune pretrained features using a small learning rate.
    base_model.trainable = True
    # Keep BatchNorm statistics stable on the comparatively small dataset.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    _compile(model, FINE_TUNE_LEARNING_RATE)
    model.fit(
        train_ds,
        validation_data=validation_ds,
        initial_epoch=HEAD_EPOCHS,
        epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
        callbacks=[tensorboard],
    )

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"TensorBoard logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
