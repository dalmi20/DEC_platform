import cv2
import numpy as np

def lab_log_preprocessing(image):
    print("method5")
   
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # Separate the lightness channel
    L_channel = lab_image[:, :, 0]

    # Apply log transformation to the lightness channel
    L_log = np.log1p(L_channel)

    # Normalize the log-transformed lightness channel to [0, 255]
    L_log_normalized = (L_log - np.min(L_log)) / (np.max(L_log) - np.min(L_log)) * 255

    # Combine the modified lightness channel with the original color channels
    lab_image[:, :, 0] = L_log_normalized.astype(np.uint8)

    # Reshape for deep embedded clustering
    x = lab_image.reshape((-1, 3))

    # Normalize the data
    x = x / 255

    return x

