# main.py

import os
from src.data_loader import load_data
from src.preprocessing import impute_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def main():
    # Set folder path where the CSVs are located
    data_path = os.path.join(os.getcwd(), "data")

    # Load training and test data
    print("📥 Loading data...")
    train_df, _ = load_data(data_path)

    # Preprocess training data
    print("🧹 Preprocessing training data...")
    train_df, label_encoders = impute_features(train_df)

    # Split features and target
    X = train_df.drop(columns=["Personality"])
    y = train_df["Personality"]

    # Train the model
    print("🧠 Training model...")
    pipeline, _ = train_model(X, y)  # Saves model.pkl and labelEncoder.pkl

    # Evaluate and save metrics
    print("📊 Evaluating model...")
    evaluate_model(pipeline, X, y)

    print("✅ Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
