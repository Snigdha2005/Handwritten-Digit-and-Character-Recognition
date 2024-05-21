from image_processing import image_adjustment
import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

input_dir = 'subashini_handwriting'
output_dir = 'subashini_adjusted_letters'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg')):
        image_adjustment(input_dir, filename, output_dir)
