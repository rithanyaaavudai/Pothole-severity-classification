import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Dataset paths
train_dir = "/content/pothole_data/train"
validation_dir = "/content/pothole_data/validation"
test_dir = "/content/pothole_data/test"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Pre-trained CNN
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Build model
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# Test accuracy
test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)

# Predictions
predictions = []
actual_labels = []

for images, labels in test_data:
    output = model.predict(images, verbose=0)
    predicted = np.argmax(output, axis=1)

    predictions.extend(predicted)
    actual_labels.extend(labels.numpy())

predictions = np.array(predictions)
actual_labels = np.array(actual_labels)

# Sample prediction
sample_image = next(iter(test_data))[0][0:1]

prediction = model.predict(sample_image, verbose=0)
predicted_class = np.argmax(prediction)

print("Predicted Severity:", class_names[predicted_class])

# Accuracy graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()

# Loss graph
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# Actual vs Predicted
plt.figure(figsize=(10, 5))

plt.plot(
    actual_labels[:30],
    marker="o",
    label="Actual"
)

plt.plot(
    predictions[:30],
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted Pothole Severity")
plt.xlabel("Test Image")
plt.ylabel("Severity Class")
plt.legend()
plt.grid(True)

plt.show()

