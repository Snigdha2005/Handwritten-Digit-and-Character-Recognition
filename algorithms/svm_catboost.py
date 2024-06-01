import numpy as np
from Preprocessing.preprocessing import preprocessed_images, labels
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

# Flatten images for non-CNN models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)

# Import models for stacking
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
catboost = CatBoostClassifier(iterations=100, random_state=42, verbose=0)

# Define meta-model
meta_model = LogisticRegression()

# Define stacking classifier
stacking_classifier = StackingClassifier(
    estimators=[('svm', svm_rbf), ('catboost', catboost)],
    final_estimator=meta_model,
    passthrough=True,  # Use original features along with base models' predictions
    cv=5
)

# Train stacking classifier
stacking_classifier.fit(X_train_flat, y_train_flat)

# Predict
y_pred_stacking = stacking_classifier.predict(X_test_flat)

# Evaluate
accuracy_stacking = accuracy_score(y_test_flat, y_pred_stacking)
precision_stacking = precision_score(y_test_flat, y_pred_stacking, average='weighted')
print(f"Stacking Classifier Accuracy: {accuracy_stacking}")
print(f"Stacking Classifier Precision: {precision_stacking}")
print(classification_report(y_test_flat, y_pred_stacking))
