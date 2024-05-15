from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from thicken import thicken_letters

def image_adjustment1(image_name):
    # Load image
    image_path = 'snigdha_handwriting/'+image_name
    #image = Image.open(image_path)

    # Adjust contrast
    #enhancer = ImageEnhance.Contrast(image)
    #image = enhancer.enhance(1)  # Low contrast
    # Adjust brightness to simulate pop effect
    #enhancer = ImageEnhance.Brightness(image)
    #image = enhancer.enhance(0.5)  # High brightness

    # Adjust sharpness
    #enhancer = ImageEnhance.Sharpness(image)
    #image = enhancer.enhance(6.0)  # High sharpness

    # Convert to OpenCV image (numpy array) for further processing
    #image_cv = np.array(image)

    # Denoise
    #dst = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
    # Convert to black and white
    #(thresh, blackAndWhiteImage) = cv2.threshold(image_cv, 600, 255, cv2.THRESH_BINARY)

    # Adjust white point (brightness)
    #image_cv = cv2.convertScaleAbs(dst, alpha=0.3, beta=20)

    # Adjust black point (contrast)
    #image_cv = cv2.convertScaleAbs(image_cv, alpha=6, beta=0)

    # Save the image
    #cv2.imwrite('snigdha_adjusted_handwriting/'+image_name[0]+'_adjusted_image.jpg', image_cv)
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 3)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 0, 0), 5)

    # Save the image
    cv2.imwrite('snigdha_adjusted_handwriting/'+image_name[0]+'_adjusted_image.jpg', image)

def image_adjustment(image_name):
    image_path = 'pen_letters/'+image_name
    image = cv2.imread(image_path)
    with Image.open(image_path) as img:
        bw = img.convert('L')
        enhancer = ImageEnhance.Contrast(bw)
        pop_img = enhancer.enhance(10)
        pop_img_np = np.array(pop_img)
        pop_img.save('pen_letters/'+image_name[0]+'_adjusted_image.jpg')
        #edges = cv2.Canny(pop_img_np, 30, 100)
        #thicken_letters('snigdha_adjusted_handwriting/'+image_name[0]+'_adjusted_image.jpg')
        # Display the original image and the edges side by side
        cv2.imshow('Original Image', image)
        cv2.imshow('Edited Image', pop_img_np)
        #cv2.imshow('Edges', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
