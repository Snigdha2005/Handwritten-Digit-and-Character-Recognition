from lightgbm import LGBMClassifier
from Preprocessing.data_prep import X_test_flat, y_train_flat, y_test_flat, X_train_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train LightGBM
lgbm = LGBMClassifier(n_estimators=500, random_state=42)
lgbm.fit(X_train_flat, y_train_flat)

# Predict
y_pred_lgbm = lgbm.predict(X_test_flat)

# Evaluate
accuracy_lgbm = accuracy_score(y_test_flat, y_pred_lgbm)
precision_lgbm = precision_score(y_test_flat, y_pred_lgbm, average='weighted')
print(f"LightGBM Accuracy: {accuracy_lgbm}")
print(f"LightGBM Precision: {precision_lgbm}")
print(classification_report(y_test_flat, y_pred_lgbm))
