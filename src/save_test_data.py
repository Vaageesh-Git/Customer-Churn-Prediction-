"""
One-time script to generate and save the test split for evaluation.
Replicates the exact preprocessing and split used during model training
so that metrics computed here match the notebook's reported values.

Run from repo root: python -m src.save_test_data
Outputs: models/test_data.pkl
"""

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import load_columns, load_scaler


def generate_test_data(
    csv_path: str = "data/Telco-Customer-Churn.csv",
    output_path: str = "models/test_data.pkl",
):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Replicate notebook preprocessing exactly
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Drop non-feature columns
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encode and align to training columns
    X_dummies = pd.get_dummies(X)
    training_columns = load_columns()
    X_aligned = X_dummies.reindex(columns=training_columns, fill_value=0)

    # Same split as notebook: test_size=0.2, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X_aligned, y, test_size=0.2, random_state=42
    )

    # Scale using the saved scaler (fit on SMOTE-balanced training data)
    scaler = load_scaler()
    X_test_scaled = scaler.transform(X_test)

    payload = {
        "X_test_scaled": X_test_scaled,
        "y_test": y_test.values,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved test data to {output_path}")
    print(f"  Test set size: {len(y_test)} samples")
    print(f"  Churn rate in test set: {y_test.mean():.1%}")


if __name__ == "__main__":
    generate_test_data()
