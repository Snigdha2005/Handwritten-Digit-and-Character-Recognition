import cv2
import numpy as np
from PIL import Image, ImageEnhance

def add_white_corner_mask(image, mask_size=1):
    # Create a white mask
    mask = np.ones_like(image) * 255

    # Define corner coordinates for the mask
    height, width = image.shape
    mask[0:mask_size, 0:mask_size] = 255
    mask[0:mask_size, width-mask_size:width] = 255
    mask[height-mask_size:height, 0:mask_size] = 255
    mask[height-mask_size:height, width-mask_size:width] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_or(image, mask)
    return masked_image

def thicken_letters(image_name, kernel_size=3, iterations=1):
    image_path = image_name
    
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply contrast enhancement if needed
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2)  # Increase contrast
    image = np.array(enhanced_image)
    masked_image = image
    # Apply binary thresholding
    _, binary_image = cv2.threshold(masked_image, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Define a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation to thicken the letters/digits
    thickened_image = cv2.dilate(binary_image, kernel, iterations=iterations)
    
    # Invert the image back
    thickened_image = cv2.bitwise_not(thickened_image)
    # Save the result
    result_image_path = 'snigdha_adjusted_handwriting/' + image_name[0] + '_thickened.jpg'
    cv2.imwrite(result_image_path, thickened_image)
    
    # Display the original and thickened images
    #cv2.imshow('Original Image', image)
    #cv2.imshow('Thickened Image', thickened_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
