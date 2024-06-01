from sklearn.svm import SVC
from Preprocessing.data_prep import le, X_test_flat, X_train_flat, y_train_flat, y_test_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_flat, y_train_flat)

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test_flat)
print("SVM Accuracy:", accuracy_score(y_test_flat, y_pred_svm))
print("SVM Precision:", precision_score(y_test_flat, y_pred_svm, average='weighted'))
print("Classification Report:\n", classification_report(y_test_flat, y_pred_svm, target_names=le.classes_))
