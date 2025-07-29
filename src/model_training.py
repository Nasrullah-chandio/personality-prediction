import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def train_model(X, y):
    label_encoders = {}

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Define CatBoost model
    model = CatBoostClassifier(
        iterations=200,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=5,
        random_strength=3,
        verbose=0,
        random_state=42
    )

    # Build pipeline with SMOTE
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    # Train pipeline
    pipeline.fit(X, y)

    # Evaluate on training data
    y_pred_train = pipeline.predict(X)
    train_acc = accuracy_score(y, y_pred_train)
    print(f"Model trained | Training Accuracy: {train_acc:.4f}")

    # Save pipeline and label encoders
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, "model.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))

    print(f"Saved model and encoders to: {model_dir}/")

    return pipeline, label_encoders
