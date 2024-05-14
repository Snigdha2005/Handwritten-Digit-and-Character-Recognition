import os
import zipfile
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

df = pd.read_csv('data/english.csv')
image_paths = ['data/img/' + img_path for img_path in df['image']]
labels = df['label']

print(df.head())
print("First few image paths:")
print(image_paths[:5])
print("First few labels:")
print(labels[:5])

def preprocess_image(image_path):
    image_name = os.path.basename(image_path)
    full_image_path = os.path.join('data', 'Img', image_name)

    try:
        img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception(f"Failed to load image at path '{full_image_path}'")

        img = cv2.resize(img, (28, 28))
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        print(f"Error processing image at path '{full_image_path}': {e}")
        return None

preprocessed_images = []

for path in image_paths:
    img = preprocess_image(path)
    if img is not None:
        preprocessed_images.append(img)

print(f"Number of successfully preprocessed images: {len(preprocessed_images)}")

