import numpy as np
from preprocessing import preprocessed_images, labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, classification_report
from tensorflow.keras.utils import to_categorical

# Convert list of images to numpy array
X = np.array(preprocessed_images)
# Reshape data to match model input requirements
X = X.reshape(-1, 28, 28, 1)  # for CNN

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)  # one-hot encode for CNN

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten images for Logistic Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)


