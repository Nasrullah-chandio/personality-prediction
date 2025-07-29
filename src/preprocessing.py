import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def impute_features(df, is_train=True, label_encoder=None):
    # Drop 'id' column if it exists
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # If training, we expect 'Personality' column to exist and guide imputation
    if is_train:
        def impute_drained(row):
            if pd.isna(row["Drained_after_socializing"]):
                return "No" if row["Personality"] == "Extrovert" else "Yes"
            return row["Drained_after_socializing"]

        def impute_stage_fear(row):
            if pd.isna(row["Stage_fear"]):
                return "No" if row["Personality"] == "Extrovert" else "Yes"
            return row["Stage_fear"]

        df["Drained_after_socializing"] = df.apply(impute_drained, axis=1)
        df["Stage_fear"] = df.apply(impute_stage_fear, axis=1)

        median_map = {
            'Time_spent_Alone': df.groupby('Personality')["Time_spent_Alone"].median(),
            'Post_frequency': {'Extrovert': 6.0, 'Introvert': 1.0},
            'Social_event_attendance': {'Extrovert': 6.0, 'Introvert': 2.0},
            'Friends_circle_size': {'Extrovert': 10.0, 'Introvert': 3.0},
            'Going_outside': {'Extrovert': 5.0, 'Introvert': 2.0}
        }

        for col, values in median_map.items():
            if isinstance(values, dict):
                df[col] = df.apply(lambda row: values[row['Personality']] if pd.isna(row[col]) else row[col], axis=1)
            else:
                df[col] = df.apply(lambda row: values[row['Personality']] if pd.isna(row[col]) else row[col], axis=1)

        df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
        df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})

        # Encode target
        label_encoder = LabelEncoder()
        df["Personality"] = label_encoder.fit_transform(df["Personality"])

        return df, label_encoder

    else:
        # For inference or test data (no Personality-based imputation)
        df["Stage_fear"] = df["Stage_fear"].map({"Yes": 1, "No": 0})
        df["Drained_after_socializing"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})

        return df, label_encoder

