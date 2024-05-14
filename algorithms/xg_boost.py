import xgboost as xgb
from data_prep import le, X_test_flat, X_train_flat, y_train, y_test
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_flat, y_train)

# Evaluate XGBoost model
y_pred_xgb = xgb_model.predict(X_test_flat)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=le.classes_))
