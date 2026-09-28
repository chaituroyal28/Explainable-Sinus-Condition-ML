"""
Machine learning models for the Sinus Condition prediction project.
"""

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def create_models(preprocessor):

    models = {

        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    random_state=42
                )
            )
        ]),

        "SVM": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42
                )
            )
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss"
                )
            )
        ]),

        "CatBoost": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                CatBoostClassifier(
                    verbose=0,
                    random_state=42
                )
            )
        ])
    }

    return models


def create_parameter_distributions():

    parameter_distributions = {

        "Logistic Regression": {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"]
        },

        "Decision Tree": {
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy", "log_loss"]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 8, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5]
        },

        "SVM": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"]
        },

        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0]
        },

        "CatBoost": {
            "model__iterations": [100, 200, 300],
            "model__depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.05, 0.1]
        }
    }

    return parameter_distributions