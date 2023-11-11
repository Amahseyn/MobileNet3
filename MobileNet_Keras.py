import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define constants
input_shape = (224, 224, 1)  # Grayscale images have one channel
num_classes = 1  # Binary segmentation

# Data paths
train_data_dir = 'path/to/train'
val_data_dir = 'path/to/validation'
test_data_dir = 'path/to/test'

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

def preprocess_mask(mask):
    # Apply thresholding (adjust threshold value as needed)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Apply noise reduction (adjust kernel size as needed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))

    return mask / 255.0  # Normalize to [0, 1]

def preprocess_input(image):
    # No preprocessing for images, just rescale
    return image / 255.0

def visualize_data(generator, num_samples=5):
    for i in range(num_samples):
        sample = next(generator)
        image, mask = sample[0][0], sample[1][0]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask[:, :, 0], cmap='viridis')
        plt.title('Mask')
        plt.show()

# Visualize some training data
visualize_data(train_generator)

# Base MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the segmentation model
x = base_model.output
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D()(x)
output = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
evaluation = model.evaluate(test_generator, steps=test_generator.n // batch_size)
print(f'Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

# Make predictions on the test set
predictions = model.predict(test_generator, steps=test_generator.n // batch_size)

# Visualize some predictions (optional)
for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_generator[i][0][0], cmap='gray')
    plt.title('Input Image')
    plt.subplot(1, 2, 2)
    plt.imshow(predictions[i, :, :, 0], cmap='viridis')
    plt.title('Predicted Mask')
    plt.show()
