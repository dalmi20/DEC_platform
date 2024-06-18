import numpy as np 
from keras.initializers import VarianceScaling
from keras.optimizers import SGD
import cv2
import os
import numpy as np
 
def skull_stripping(arr): 
# Step 1: Apply median filtering with a window of size 3 × 3
    def apply_median_filter(image):
        return cv2.medianBlur(image, 3)
    
    # Step 2: Compute the initial mean intensity value Ti of the image
    def compute_initial_mean_intensity(image):
        return np.mean(image)
    
    # Step 3: Identify the top, bottom, left, and right pixel locations of the skull
    def identify_skull_boundaries(image, Ti):
        skull_indices = np.where(image > Ti)
        top, bottom = np.min(skull_indices[0]), np.max(skull_indices[0])
        left, right = np.min(skull_indices[1]), np.max(skull_indices[1])
        return top, bottom, left, right
    
    # Step 4: Form a rectangle using the top, bottom, left, and right pixel locations
    # Step 5: Compute the final mean value Tf of the brain using the pixels located within the rectangle
    def compute_final_mean_brain_intensity(image, top, bottom, left, right):
        brain_region = image[top:bottom+1, left:right+1]
        return np.mean(brain_region)
    
    # Step 6: Approximate the region of brain membrane based on intensity comparison
    # Step 7: Set the average intensity value of membrane as the threshold value T
    def compute_threshold_value(image, Tf, skull_top, skull_bottom, skull_left, skull_right):
        membrane_region = image[skull_top:skull_bottom+1, skull_left:skull_right+1]
        return np.mean(membrane_region)
    
    # Step 8: Convert the given input image into binary image using the threshold T
    def convert_to_binary(image, T):
        _, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)
        return binary_image
    
    # Step 9: Apply a 13 × 13 opening morphological operation to separate the skull from the brain
    def apply_morphological_operations(binary_image):
        kernel_opening = np.ones((13, 13), np.uint8)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_opening)
        return opening
    
    # Step 10: Find the largest connected component and consider it as brain
    def find_largest_connected_component(image):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        brain_mask = np.uint8(labels == largest_label) * 255
        return brain_mask
    
    # Step 11: Finally, apply a 21 × 21 closing morphological operation to fill the gaps within and along the periphery of the intracranial region
    def fill_gaps(image):
        kernel_closing = np.ones((21, 21), np.uint8)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_closing)
        return closing
    print('typprrrr',type(arr))
    array = np.array(arr,dtype=np.uint8)
    image = cv2.cvtColor(array,cv2.COLOR_BGR2GRAY)
    print('ok')
    # Step 1: Apply median filtering
    filtered_image = apply_median_filter(image)
    
    # Step 2: Compute initial mean intensity value
    Ti = 44
    
    # Step 3: Identify skull boundaries
    skull_top, skull_bottom, skull_left, skull_right = identify_skull_boundaries(filtered_image, Ti)
    
    # Step 4 & 5: Compute final mean value of the brain intensity
    Tf = 80
    
    # Step 6 & 7: Compute threshold value
    T = 38
    
    # Step 8: Convert image to binary
    binary_image = convert_to_binary(filtered_image, T)
    
    # Step 9: Apply morphological operations
    morphological_image = apply_morphological_operations(binary_image)
    
    # Step 10: Find largest connected component
    brain_mask = find_largest_connected_component(morphological_image)
    
    # Step 11: Fill gaps within and along the periphery of the intracranial region
    filled_brain = fill_gaps(brain_mask)
    
    brain_image = cv2.bitwise_and(image,image, mask=filled_brain)
    brain_image = cv2.cvtColor(brain_image , cv2.COLOR_GRAY2RGB)
    
    return brain_image