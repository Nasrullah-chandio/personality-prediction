import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def evaluate_model(model, X_test, y_true):
    """
    Evaluates the trained model on the test set and saves evaluation metrics.

    Parameters:
        model: Trained model pipeline
        X_test: Test features
        y_true: True labels
    """
    # Drop 'id' column if it exists
    if "id" in X_test.columns:
        X_test = X_test.drop(columns=["id"])

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    conf_matrix = confusion_matrix(y_true, y_pred).tolist()
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Save metrics to JSON
    metrics = {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision
    }

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("Model evaluation completed.")
    print("Metrics saved to artifacts/evaluation_metrics.json")
    print(f"Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
