"""
SinusPredict AI
Flask web application.
"""

from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from explainability import explain_prediction


# =========================================================
# APPLICATION SETUP
# =========================================================

APP_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "best_sinus_condition_model.pkl"
)

model = joblib.load(MODEL_PATH)


app = Flask(
    __name__,
    template_folder=str(
        APP_DIR / "templates"
    ),
    static_folder=str(
        APP_DIR / "static"
    )
)


# =========================================================
# PAGE ROUTES
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/analyze")
def analyze():

    return render_template(
        "analyze.html"
    )


@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html"
    )


@app.route("/insights")
def insights():

    return render_template(
        "insights.html"
    )


@app.route("/models")
def models():

    return render_template(
        "models.html"
    )


@app.route("/research")
def research():

    return render_template(
        "research.html"
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================================
# PREDICTION API
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        input_data = {

            "Age":
                int(data["Age"]),

            "Gender":
                data["Gender"],

            "Regular Sinus Issues":
                data[
                    "Regular Sinus Issues"
                ],

            "Most Problematic Season":
                data[
                    "Most Problematic Season"
                ],

            "Triggered by External Factors":
                data[
                    "Triggered by External Factors"
                ],

            "Known Allergy":
                data[
                    "Known Allergy"
                ],

            "Doctor Consulted (Past Year)":
                data[
                    "Doctor Consulted (Past Year)"
                ],

            "Sinus Severity (1-5)":
                int(
                    data[
                        "Sinus Severity (1-5)"
                    ]
                ),

            "Headaches or Facial Pain":
                data[
                    "Headaches or Facial Pain"
                ],

            "Medication Frequency":
                data[
                    "Medication Frequency"
                ]

        }

        input_df = pd.DataFrame(
            [input_data]
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = list(
            model.classes_
        )

        probability_map = dict(
            zip(
                classes,
                probabilities
            )
        )

        predicted_probability = float(
            probability_map.get(
                prediction,
                max(probabilities)
            )
        )

        prediction_label = (
            "Yes"
            if int(prediction) == 1
            else "No"
        )


        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        explanation = explain_prediction(
            model,
            input_df,
            top_n=8
        )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "prediction":
                prediction_label,

            "probability":
                round(
                    predicted_probability * 100,
                    2
                ),

            "model":
                "Tuned Decision Tree",

            "test_f1":
                56.90,

            "test_roc_auc":
                53.68,

            "inputs":
                input_data,

            "explanation":
                explanation

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )