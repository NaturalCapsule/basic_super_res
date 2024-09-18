import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers

# Function to create low-resolution images
def create_lr_images(hr_folder, lr_folder, scale_factor=4):
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    for filename in os.listdir(hr_folder):
        img_path = os.path.join(hr_folder, filename)
        hr_image = cv2.imread(img_path)
        
        if hr_image is not None:
            height, width, _ = hr_image.shape
            new_height, new_width = height // scale_factor, width // scale_factor
            
            # Resize to low-resolution
            lr_image = cv2.resize(hr_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            lr_image_upscaled = cv2.resize(lr_image, (width, height), interpolation=cv2.INTER_LINEAR)
            
            lr_image_path = os.path.join(lr_folder, filename)
            cv2.imwrite(lr_image_path, lr_image_upscaled)
            print(f"Saved low-resolution image: {lr_image_path}")

# Function to load all images from a folder with consistent resizing
def load_and_resize_images(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize images to the target size (e.g., 128x128)
            img_resized = cv2.resize(img, target_size)
            # Normalize the image to 0-1 for the model
            img_resized = img_resized.astype(np.float32) / 255.0
            images.append(img_resized)
    return images

# Build SRCNN model
def build_srcnn():
    model = tf.keras.Sequential()
    
    # First convolutional layer
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(128, 128, 3)))
    
    # Second convolutional layer
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    
    # Output layer
    model.add(layers.Conv2D(3, (5, 5), activation='linear', padding='same'))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def train_model(model, lr_images, hr_images, epochs=100, batch_size=16):
    model.fit(np.array(lr_images), np.array(hr_images), epochs=epochs, batch_size=batch_size, validation_split=0.1)

def super_resolve(model, lr_image):
    lr_image = np.expand_dims(lr_image, axis=0)  # Add batch dimension
    sr_image = model.predict(lr_image)
    return sr_image[0]  # Remove batch dimension

hr_folder = 'Set14/Set14/'  # Folder containing high-resolution images
lr_folder = 'lr_res/'       # Folder where low-resolution images will be saved
create_lr_images(hr_folder, lr_folder, scale_factor=4)

target_size = (128, 128)  # All images resized to this size for consistent training
lr_images = load_and_resize_images(lr_folder, target_size=target_size)
hr_images = load_and_resize_images(hr_folder, target_size=target_size)

model = build_srcnn()

train_model(model, lr_images, hr_images, epochs=100, batch_size=16)

sr_image = super_resolve(model, lr_images[0])  # Pass a single image

sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)
cv2.imwrite('super_resolved.jpg', sr_image)
print("Super-resolved image saved.")
