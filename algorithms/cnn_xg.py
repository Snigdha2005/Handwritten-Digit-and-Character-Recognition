import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from Preprocessing.data_prep import le, X_train, y_train, X_test, y_test
from tensorflow.keras.layers import Input

# Corrected one-hot encoding (if necessary)
# y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=number_of_classes)
# y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=number_of_classes)

# Build CNN model with Input layer
cnn_model = Sequential([
    Input(shape=(28, 28, 1)),  # Use Input layer to define input shape
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')  # Ensure this matches the number of classes
])

# Rest of your code remains the same...

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train, y_train_one_hot, epochs=15, batch_size=32)

# Extract features from CNN
cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
X_train_cnn_features = cnn_feature_extractor.predict(X_train)
X_test_cnn_features = cnn_feature_extractor.predict(X_test)

# Train XGBoost model using CNN features
xgb_model_cnn = XGBClassifier()
xgb_model_cnn.fit(X_train_cnn_features, y_train)

# Evaluate XGBoost model using CNN features
y_pred_xgb = xgb_model_cnn.predict(X_test_cnn_features)
xgb_accuracy_cnn = accuracy_score(y_test, y_pred_xgb)
print("Combined Model (CNN + XGBoost) Accuracy:", xgb_accuracy_cnn)
