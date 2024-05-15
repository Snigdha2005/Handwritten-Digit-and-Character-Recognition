import cv2
import numpy as np
from PIL import Image, ImageEnhance

def create_corner_mask(image_shape, mask_size=50):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Top-left corner
    mask[:mask_size, :mask_size] = 255
    mask = cv2.GaussianBlur(mask, (mask_size*2+1, mask_size*2+1), 0)
    
    # Top-right corner
    mask[:mask_size, -mask_size:] = 255
    mask = cv2.GaussianBlur(mask, (mask_size*2+1, mask_size*2+1), 0)
    
    # Bottom-left corner
    mask[-mask_size:, :mask_size] = 255
    mask = cv2.GaussianBlur(mask, (mask_size*2+1, mask_size*2+1), 0)
    
    # Bottom-right corner
    mask[-mask_size:, -mask_size:] = 255
    mask = cv2.GaussianBlur(mask, (mask_size*2+1, mask_size*2+1), 0)
    
    return mask

def image_adjustment(image_name):
    image_path = 'snigdha_handwriting/' + image_name
    with Image.open(image_path) as img:
        # Convert to grayscale and enhance contrast
        bw = img.convert('L')
        enhancer = ImageEnhance.Contrast(bw)
        pop_img = enhancer.enhance(10)
        pop_img_np = np.array(pop_img)
        
        # Create a mask with white corners
        mask = create_corner_mask(pop_img_np.shape, mask_size=100)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Convert the pop_img to a format compatible with cv2
        pop_img_cv2 = cv2.cvtColor(pop_img_np, cv2.COLOR_GRAY2RGB)
        
        # Blend the mask with the image
        blended_np = cv2.addWeighted(pop_img_cv2, 1.0, mask_rgb, 0.5, 0)
        blended = Image.fromarray(blended_np)
        # Save the result
        result_image_path = 'snigdha_adjusted_handwriting/' + image_name[0] + '_adjusted_image.jpg'
        blended.save(result_image_path)
        
        # Display the original image and the blended image side by side
        #cv2.imshow('Original Image', pop_img_cv2)
        #cv2.imshow('Blended Image', blended_np)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
