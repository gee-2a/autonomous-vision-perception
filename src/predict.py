from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt


MODEL_PATH = Path("../outputs/traffic_sign_classifier.keras")
IMAGE_PATH = Path("images/img_1.jpeg")  # replace with an image path

model = tf.keras.models.load_model(MODEL_PATH)

image = tf.io.read_file(str(IMAGE_PATH))
image = tf.io.decode_image(image, channels=3, expand_animations=False)
image = tf.image.resize(image, (224, 224))
image = tf.keras.applications.mobilenet_v2.preprocess_input(
    tf.cast(image, tf.float32)
)

probabilities = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
class_id = int(tf.argmax(probabilities))
confidence = float(tf.reduce_max(probabilities))

print(f"Predicted class: {class_id}")
print(f"Confidence: {confidence:.1%}")


plt.imshow(tf.io.decode_image(
    tf.io.read_file(str(IMAGE_PATH)), channels=3
).numpy())
plt.title(f"Class {class_id} — {confidence:.1%} confidence")
plt.axis("off")
plt.show()