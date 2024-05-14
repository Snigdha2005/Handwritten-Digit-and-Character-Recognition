import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from data_prep import le, X_train, y_train, X_test, y_test

# One-hot encode the labels
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(le.classes_))
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(le.classes_))

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

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=15, batch_size=32)

# Extract features from CNN
cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
X_train_cnn_features = cnn_feature_extractor.predict(X_train)
X_test_cnn_features = cnn_feature_extractor.predict(X_test)

# Train GBM model using CNN features
gbm_model_cnn = GradientBoostingClassifier()
gbm_model_cnn.fit(X_train_cnn_features, y_train)

# Evaluate GBM model using CNN features
y_pred_gbm = gbm_model_cnn.predict(X_test_cnn_features)
gbm_accuracy_cnn = accuracy_score(y_test, y_pred_gbm)
print("Combined Model (CNN + GBM) Accuracy:", gbm_accuracy_cnn)
