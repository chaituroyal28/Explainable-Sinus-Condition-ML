"""
Evaluation utilities for the Sinus Condition ML project.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """Calculate standard binary classification metrics."""

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)

    else:
        probabilities = predictions

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }


def confusion_matrix_data(model, X_test, y_test):
    """Return confusion matrix."""

    predictions = model.predict(X_test)

    return confusion_matrix(
        y_test,
        predictions
    )


def classification_report_data(model, X_test, y_test):
    """Return classification report as dictionary."""

    predictions = model.predict(X_test)

    return classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]

    return results_df[columns]