import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from cnn import cnn_model
from data_prep import le  # Assuming le is your LabelEncoder

# Load the image
img = load_img('A_image.jpg', color_mode='grayscale', target_size=(28, 28))

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

print("Predicted label:", predicted_label, end = " ")
print("Predicted class:", predicted_class)
