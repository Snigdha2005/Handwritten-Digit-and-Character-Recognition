import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from CNN.cnn import cnn_model
from Preprocessing.data_prep import le  # Assuming le is your LabelEncoder
import sys
import os

# Directory containing the images
directory = 'pen_adjusted_letters'
correct_labels = []
wrong_labels = []
# Get a list of all the image filenames in the directory
image_filenames = os.listdir(directory)

# Loop over each image
for filename in image_filenames:
    # Load the image
    img = load_img(os.path.join(directory, filename), color_mode='grayscale', target_size=(28, 28))

    # Convert the image to an array
    img_array = img_to_array(img)

    # Reshape the array for the model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Normalize the image (if your model expects normalized input)
    img_array = img_array / 255.0

    # Use the model to make a prediction
    prediction = cnn_model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_class])

    # Get the first letter of the image filename
    true_label = filename[0]

    print("Image:", filename, end = " ")
    print("True label:", true_label, end = " ")
    print("Predicted label:", predicted_label, end = " ")
    print("Predicted class:", predicted_class)

    # Check if the prediction is correct
    if predicted_label == true_label:
        correct_labels.append(true_label)
    else:
        wrong_labels.append(true_label)

print("Number of correct predictions:", len(correct_labels), correct_labels)
print("Number of wrong predictions:", len(wrong_labels), wrong_labels)
