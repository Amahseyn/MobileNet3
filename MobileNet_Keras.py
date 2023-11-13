import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Define constants
input_shape = (224, 224, 3)  # Change the number of channels to 3 for RGB
num_classes = 1  # Binary segmentation

# Data paths
train_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/train'
val_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/val'
test_data_dir = '/content/drive/MyDrive/MobileNet/DataSet/test'

def load_and_preprocess_data(data_dir):
    images = []
    masks = []

    for image_filename in os.listdir(os.path.join(data_dir, 'images')):
        image_path = os.path.join(data_dir, 'images', image_filename)
        mask_path = os.path.join(data_dir, 'masks', image_filename)
        
        # Read and preprocess image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (input_shape[0], input_shape[1]))
        image = image / 255.0
        
        # Read and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask >= 10).astype(np.float32)  # Ensure the correct data type
        mask = cv2.resize(mask, (input_shape[0], input_shape[1]))
        mask = mask / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)

def visualize_data(images, masks, num_samples=5):
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(masks[i][:, :, 0], cmap='viridis')
        plt.title('Mask')
        plt.show()

# Load and preprocess training data
train_images, train_masks = load_and_preprocess_data(train_data_dir)

# Load and preprocess validation data
val_images, val_masks = load_and_preprocess_data(val_data_dir)

# Load and preprocess test data
test_images, test_masks = load_and_preprocess_data(test_data_dir)

# Visualize some training data
# visualize_data(train_images, train_masks)

# Base MobileNetV3 model
base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the segmentation model
x = base_model.output
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(4, 4))(x)  # Upsample to match 224x224
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D(size=(2, 2))(x)
output = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
epochs = 1000
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=epochs
)


model.save("1.h5")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
evaluation = model.evaluate(test_images, test_masks)
print(f'Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}')

# Make predictions on the test set
predictions = model.predict(test_images)

# Visualize some predictions (optional)
for i in range(5):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(test_images[i])
    plt.title('Input Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(test_masks[i][:, :, 0], cmap='viridis')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i] > 0.5, cmap='viridis')  # Apply threshold
    plt.title('Predicted Mask')
    plt.show()
