import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from data_prep import le, X_test, X_train, y_train, y_test
import numpy as np

# Define the CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=32)

# Ensure the model is built by calling it with some input data

cnn_model.build((None, 28, 28, 1))

# Extract features from the CNN
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

# Get the features for the training and test data
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# Flatten the labels for SVM
y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, y_train_flat)

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test_features)
print("SVM Accuracy:", accuracy_score(y_test_flat, y_pred_svm))
print("Classification Report:\n", classification_report(y_test_flat, y_pred_svm, target_names=le.classes_))
