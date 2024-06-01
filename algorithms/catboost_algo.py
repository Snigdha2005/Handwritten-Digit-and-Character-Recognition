from catboost import CatBoostClassifier
from Preprocessing.data_prep import X_train_flat, y_train_flat, X_test_flat, y_test_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train CatBoost
catboost = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
catboost.fit(X_train_flat, y_train_flat)

# Predict
y_pred_catboost = catboost.predict(X_test_flat)

# Evaluate
accuracy_catboost = accuracy_score(y_test_flat, y_pred_catboost)
precision_catboost = precision_score(y_test_flat, y_pred_catboost, average='weighted')
print(f"CatBoost Accuracy: {accuracy_catboost}")
print(f"CatBoost Precision: {precision_catboost}")
print(classification_report(y_test_flat, y_pred_catboost))
