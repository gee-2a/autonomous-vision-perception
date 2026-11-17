"""Transfer-learning model for 43 German traffic sign classes."""
import tensorflow as tf

from config import IMAGE_SIZE, NUM_CLASSES


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")

    # On-the-fly augmentation helps the classifier generalize to shifted/zoomed signs.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    x = augmentation(inputs)

    # ImageNet features provide a strong starting point for transfer learning.
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Phase 1 trains only the task-specific head.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="hidden_features")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="class_probabilities"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="gtsrb_mobilenetv2")
    return model, base_model
