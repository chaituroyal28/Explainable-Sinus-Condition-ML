"""
Exploratory Data Analysis utilities for the Sinus Condition ML project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import load_data, clean_data, TARGET_COLUMN


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def dataset_summary(df):
    """Return basic dataset information."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


def categorical_summary(df, column):
    """Return frequency and percentage for a categorical column."""
    counts = df[column].value_counts(dropna=False)
    percentages = df[column].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    return pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })


def save_bar_plot(df, column, filename):
    """Create and save a bar chart for a categorical variable."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    df[column].value_counts().plot(kind="bar")

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    # Load dataset
    df = load_data()

    # Clean duplicate rows
    df_clean = clean_data(df)

    # Dataset summary
    summary = dataset_summary(df_clean)

    print("\n===== DATASET SUMMARY =====")
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")

    # Target distribution
    print("\n===== TARGET DISTRIBUTION =====")
    print(categorical_summary(df_clean, TARGET_COLUMN))

    # Generate important EDA plots
    save_bar_plot(
        df_clean,
        TARGET_COLUMN,
        "target_distribution.png"
    )

    save_bar_plot(
        df_clean,
        "Gender",
        "gender_distribution.png"
    )

    save_bar_plot(
        df_clean,
        "Regular Sinus Issues",
        "regular_sinus_issues.png"
    )

    save_bar_plot(
        df_clean,
        "Most Problematic Season",
        "problematic_season.png"
    )

    save_bar_plot(
        df_clean,
        "Known Allergy",
        "known_allergy.png"
    )

    save_bar_plot(
        df_clean,
        "Medication Frequency",
        "medication_frequency.png"
    )

    print("\nEDA completed successfully.")