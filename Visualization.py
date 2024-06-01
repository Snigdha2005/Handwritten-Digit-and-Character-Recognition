#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from data_prep import le, X_train, y_train, X_test, y_test
# Build CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

# Compile model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)

# Evaluate model
cnn_eval = cnn_model.evaluate(X_test, y_test)
print("CNN Accuracy:", cnn_eval[1])


# In[13]:


# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# # from cnn import cnn_model
# from data_prep import le  # Assuming le is your LabelEncoder

# # Load the image
# img = load_img('4.jpeg', color_mode='grayscale', target_size=(28, 28))

# # Convert the image to an array
# img_array = img_to_array(img)

# # Reshape the array for the model
# img_array = img_array.reshape(1, 28, 28, 1)

# # Normalize the image (if your model expects normalized input)
# img_array = img_array / 255.0

# # Use the model to make a prediction
# prediction = cnn_model.predict(img_array)

# # Get the predicted class
# predicted_class = np.argmax(prediction)
# predicted_label = le.inverse_transform([predicted_class])

# print("Predicted label:", predicted_label, end = " ")
# print("Predicted class:", predicted_class)


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from data_prep import le  # Assuming le is your LabelEncoder

# Read the image
img = cv2.imread('z.jpeg', 0)

# Apply binary thresholding to enhance the image
ret, thresh1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# Resize the image to match model input size
resized_img = cv2.resize(thresh1, (28, 28))

# Convert the enhanced image to an array
img_array = img_to_array(resized_img)

# Reshape the array for the model
img_array = img_array.reshape(1, 28, 28, 1)

# Normalize the image (if your model expects normalized input)
img_array = img_array / 255.0

# Use the model to make a prediction
prediction = cnn_model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(prediction)
predicted_label = le.inverse_transform([predicted_class])

print("Predicted label:", predicted_label, end=" ")
print("Predicted class:", predicted_class)


# In[15]:


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from data_prep import le  # Assuming le is your LabelEncoder

# Path to the folder containing images
folder_path = 'ragini_handwriting'
count = 0
# Iterate through all images in the folder
for filename in os.listdir(folder_path):
    count = count + 1
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, filename), 0)

        # Apply binary thresholding to enhance the image
        ret, thresh1 = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

        # Resize the image to match model input size
        resized_img = cv2.resize(thresh1, (28, 28))

        # Convert the enhanced image to an array
        img_array = img_to_array(resized_img)

        # Reshape the array for the model
        img_array = img_array.reshape(1, 28, 28, 1)

        # Normalize the image (if your model expects normalized input)
        img_array = img_array / 255.0

        # Use the model to make a prediction
        prediction = cnn_model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(prediction)
        predicted_label = le.inverse_transform([predicted_class])

        # Print the filename, predicted label, and predicted class
        print("Filename:", filename)
        print("Predicted label:", predicted_label[0])
        print("Predicted class:", predicted_class)
        print()
print(count)


# In[17]:


import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Get predictions for test data
y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
# Convert one-hot encoded labels back to original labels
y_true = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[18]:


activation_model = Model(inputs=cnn_model.input, outputs=[layer.output for layer in cnn_model.layers if isinstance(layer, Conv2D)])

# Load an example image (replace 'a1.jpg' with the path to your test image)
img = load_img('a1.jpg', color_mode='grayscale', target_size=(28, 28))
img_array = img_to_array(img)
img_array = img_array.reshape(1, 28, 28, 1)
img_array = img_array / 255.0  # Normalize the image

# Get the activations of the convolutional layers for the example image
activations = activation_model.predict(img_array)

# Plot the activation maps
for i, activation_map in enumerate(activations):
    plt.figure(figsize=(10, 10))
    for j in range(activation_map.shape[-1]):
        plt.subplot(8, 8, j+1)
        plt.imshow(activation_map[0, :, :, j], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Activation Map for Convolutional Layer {i+1}')
    plt.show()


# In[19]:


import matplotlib.pyplot as plt

# Get the weights (kernels) of the convolutional layers
conv_layers = [layer for layer in cnn_model.layers if isinstance(layer, Conv2D)]

# Plot the kernels
for i, layer in enumerate(conv_layers):
    kernels = layer.get_weights()[0]
    num_kernels = kernels.shape[3]
    plt.figure(figsize=(10, 10))
    for j in range(num_kernels):
        kernel = kernels[:, :, 0, j]
        plt.subplot(8, 8, j+1)
        plt.imshow(kernel, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Kernels for Convolutional Layer {i+1}')
    plt.show()


# In[ ]:




