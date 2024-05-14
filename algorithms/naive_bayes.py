from sklearn.naive_bayes import GaussianNB
from data_prep import le, X_test_flat, X_train_flat, y_train_flat, y_test_flat
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_flat, y_train_flat)

# Predict and evaluate
y_pred_nb = nb_model.predict(X_test_flat)
print("Naive Bayes Accuracy:", accuracy_score(y_test_flat, y_pred_nb))
print("Classification Report:\n", classification_report(y_test_flat, y_pred_nb, target_names=le.classes_))
