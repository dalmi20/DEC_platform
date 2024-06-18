import cv2
from skimage import color

def Lab_preprocessing(image):
    print("method2")

    # Convert RGB to L*a*b*
    image_lab = color.rgb2lab(image)

    # Access L*, a*, and b* channels
    L, a, b = cv2.split(image_lab)

    # Reshape the image data to a 2D array for deep embedded clustering
    x = image_lab.reshape((-1, 3))

    # Normalize the data
    x = x / 255

    return x