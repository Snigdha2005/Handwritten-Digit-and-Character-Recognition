from sklearn.linear_model import LogisticRegression
from data_prep import le, X_test_flat, X_train_flat, y_train_flat, y_test_flat
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, classification_report


# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_flat, y_train_flat)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test_flat)
print("Logistic Regression Accuracy:", accuracy_score(y_test_flat, y_pred_lr))
print("Logistic Regression Precision:", precision_score(y_test_flat, y_pred_lr, average='weighted'))
print("Classification Report:\n", classification_report(y_test_flat, y_pred_lr, target_names=le.classes_))

