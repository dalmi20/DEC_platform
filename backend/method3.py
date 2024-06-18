import cv2
import numpy as np

def log_preprocessing(image):
    print("method3")
    # Convert to RGB
    brain_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Logarithmic pre-processing
    c = 1  # Scaling factor
    log_transformed = c * np.log1p(brain_image)

    # Rescale to the range [0, 255]
    log_transformed = (255 * (log_transformed - np.min(log_transformed)) / (np.max(log_transformed) - np.min(log_transformed))).astype(np.uint8)

    # Reshape for deep embedded clustering
    x = log_transformed.reshape((-1, 3))

    # Normalize the data
    x = x / 255

    return x

