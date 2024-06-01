from sklearn.neural_network import MLPClassifier
from Preprocessing.data_prep import le, X_test, X_train, y_train, y_test
from sklearn.metrics import accuracy_score, precision_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Define MLP model
mlp_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Assuming images are 28x28
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(le.classes_), activation='softmax')  # Number of output classes
])

# Compile the model
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
mlp_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
mlp_eval = mlp_model.evaluate(X_test, y_test)
print("MLP Accuracy:", mlp_eval[1])
