import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from cnn import cnn_model
from data_prep import le  # Assuming le is your LabelEncoder

# Directory containing the images
directory = 'snigdha_handwriting'

# Get a list of all the image filenames in the directory
image_filenames = os.listdir(directory)

# Initialize counters for correct and wrong predictions
correct_predictions = {**{chr(i): 0 for i in range(65, 91)}, **{str(i): 0 for i in range(1, 10)}}
wrong_predictions = {**{chr(i): 0 for i in range(65, 91)}, **{str(i): 0 for i in range(1, 10)}}

# Run the simulation 100 times
for _ in range(100):
    # Loop over each image
    for filename in image_filenames:
        # Get the first letter of the image filename (the true label)
        true_label = filename[0].upper()

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

        # Check if the prediction is correct
        if predicted_label == true_label:
            correct_predictions[true_label] += 1
        else:
            wrong_predictions[true_label] += 1

# Print the results
for letter in correct_predictions.keys():
    print(f"Letter: {letter}")
    print(f"Correct predictions: {correct_predictions[letter]}")
    print(f"Wrong predictions: {wrong_predictions[letter]}")
    print()
