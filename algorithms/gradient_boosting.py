from sklearn.ensemble import GradientBoostingClassifier
from data_prep import le, X_test_flat, X_train_flat, y_train_flat, y_test_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train GBM model
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_train_flat, y_train_flat)

# Evaluate GBM model
y_pred_gbm = gbm_model.predict(X_test_flat)
gbm_accuracy = accuracy_score(y_test_flat, y_pred_gbm)
print("Gradient Boosting Machines Accuracy:", gbm_accuracy)
