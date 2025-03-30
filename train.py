import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

# Build a U-Net model for depth estimation
def build_depth_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)

    # Decoder
    up2 = layers.UpSampling2D((2, 2))(bottleneck)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up2)

    up1 = layers.UpSampling2D((2, 2))(conv3)
    outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up1)

    return models.Model(inputs, outputs)

# Custom physics-inspired loss function
def physics_loss(y_true, y_pred):
    # Gradient in x-direction (vertical edges)
    grad_x_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    grad_x_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    grad_x_loss = tf.reduce_mean(tf.abs(grad_x_true - grad_x_pred))

    # Gradient in y-direction (horizontal edges)
    grad_y_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    grad_y_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    grad_y_loss = tf.reduce_mean(tf.abs(grad_y_true - grad_y_pred))

    # Combine gradient losses
    gradient_loss = grad_x_loss + grad_y_loss

    # Add MAE loss
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Total loss
    return gradient_loss + mae_loss

# Prepare the model
input_shape = (256, 256, 3)  # Modify based on your image dimensions
model = build_depth_model(input_shape)
model.compile(optimizer='adam', loss=physics_loss, metrics=['mae'])

# Train the model (dummy data for demonstration; replace with actual dataset)
train_images = np.random.rand(100, 256, 256, 3)  # Replace with your lunar image dataset
train_depth_maps = np.random.rand(100, 256, 256, 1)  # Replace with your depth maps

model.fit(train_images, train_depth_maps, epochs=10, batch_size=8)

# Load a test lunar image
test_image = cv2.imread(r'C:\Users\z00522pb\projects\Sharanya Project\robustness_analysis_tool-rat_modules\crater_3.jpg')
test_image_resized = cv2.resize(test_image, (256, 256))
test_image_normalized = test_image_resized / 255.0
test_image_input = np.expand_dims(test_image_normalized, axis=0)

# Predict depth
predicted_depth = model.predict(test_image_input)[0]

# Normalize predicted depth for visualization
predicted_depth_normalized = cv2.normalize(predicted_depth, None, alpha=0, beta=255,
                                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply a colormap for better visualization
colored_depth_map = cv2.applyColorMap(predicted_depth_normalized, cv2.COLORMAP_PLASMA)

# Display and save the results
#cv2.imshow('Predicted Depth Map', colored_depth_map)
cv2.imwrite('predicted_depth_map.jpg', colored_depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()