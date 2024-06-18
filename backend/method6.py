import cv2
def no_local_means_filter_preprocessing(image):

    # Apply Non-local Means filter
    nlm_filtered_image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert to 8-bit unsigned integers
    nlm_filtered_image_uint8 = cv2.convertScaleAbs(nlm_filtered_image)

    # Convert to RGB
    brain_image_rgb = cv2.cvtColor(nlm_filtered_image_uint8, cv2.COLOR_GRAY2RGB)

    # Reshape the image for deep embedded clustering
    x = brain_image_rgb.reshape((-1, 3))

    # Normalize pixel values
    x = x / 255.0

    return x
