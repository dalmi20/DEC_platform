import cv2
import numpy as np

def hsv_preprocessing(image):
    print("method4")
 
    # Convert the image from RGB to HSV color space
    brain_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    brain_image1 = brain_image.reshape((-1, 3))

    # Normalize the data
    x = brain_image1 / 255

    return x