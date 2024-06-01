from sklearn.ensemble import RandomForestClassifier
from Preprocessing.data_prep import X_train_flat, y_train_flat, y_test_flat, X_test_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report


# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train_flat)

# Predict
y_pred_rf = rf.predict(X_test_flat)

# Evaluate
accuracy_rf = accuracy_score(y_test_flat, y_pred_rf)
precision_rf = precision_score(y_test_flat, y_pred_rf, average='weighted')
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Precision: {precision_rf}")
print(classification_report(y_test_flat, y_pred_rf))
