"""
Preprocessing utilities for the Sinus Condition ML project.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sinus_survey_data1.csv"


# Target column
TARGET_COLUMN = "Diagnosed with Sinus Condition"


# Numerical features
NUMERIC_FEATURES = [
    "Age",
    "Sinus Severity (1-5)"
]


# Categorical features
CATEGORICAL_FEATURES = [
    "Gender",
    "Regular Sinus Issues",
    "Most Problematic Season",
    "Triggered by External Factors",
    "Known Allergy",
    "Doctor Consulted (Past Year)",
    "Headaches or Facial Pain",
    "Medication Frequency"
]


def load_data(path=DATA_PATH):
    """Load the sinus survey dataset."""
    return pd.read_csv(path)


def clean_data(df):
    """Remove duplicate observations."""
    return df.drop_duplicates().copy()


def prepare_data(df):
    """
    Prepare features and binary target.

    Yes = 1
    No = 0
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = (df[TARGET_COLUMN] == "Yes").astype(int)

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """Create a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_preprocessor():
    """Create preprocessing pipeline for numerical and categorical variables."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Remove duplicates
    df_clean = clean_data(df)

    # Prepare features and target
    X, y = prepare_data(df_clean)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Display results
    print("Original dataset:", df.shape)
    print("Cleaned dataset:", df_clean.shape)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)