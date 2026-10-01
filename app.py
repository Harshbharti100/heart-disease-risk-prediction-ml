from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_heart.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")


# ==========================================================
# LOAD MODEL, SCALER AND COLUMNS
# ==========================================================

try:

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)

    print("=" * 60)
    print("HEART DISEASE MODEL LOADED SUCCESSFULLY")
    print("=" * 60)

    print("Model:", type(model).__name__)
    print("Columns:", columns)

except Exception as e:

    model = None
    scaler = None
    columns = None

    print("=" * 60)
    print("ERROR LOADING MODEL FILES")
    print("=" * 60)

    print(e)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None or columns is None:

        return render_template(
            "index.html",
            error="Model files could not be loaded."
        )

    try:

        # ==================================================
        # GET INPUTS FROM FORM
        # ==================================================

        age = int(request.form["age"])

        sex = int(request.form["sex"])

        cp = int(request.form["cp"])

        trestbps = int(request.form["trestbps"])

        chol = int(request.form["chol"])

        fbs = int(request.form["fbs"])

        restecg = int(request.form["restecg"])

        thalach = int(request.form["thalach"])

        exang = int(request.form["exang"])

        oldpeak = float(request.form["oldpeak"])

        slope = int(request.form["slope"])

        ca = int(request.form["ca"])

        thal = int(request.form["thal"])


        # ==================================================
        # CREATE DATAFRAME
        # ==================================================

        input_data = pd.DataFrame([{

            "age": age,

            "sex": sex,

            "cp": cp,

            "trestbps": trestbps,

            "chol": chol,

            "fbs": fbs,

            "restecg": restecg,

            "thalach": thalach,

            "exang": exang,

            "oldpeak": oldpeak,

            "slope": slope,

            "ca": ca,

            "thal": thal

        }])


        # ==================================================
        # ENSURE EXACT TRAINING COLUMN ORDER
        # ==================================================

        input_data = input_data[columns]


        print("\nInput Data:")
        print(input_data)


        # ==================================================
        # SCALE INPUT
        # ==================================================

        input_scaled = scaler.transform(input_data)


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(input_scaled)[0]


        print("Prediction:", prediction)


        # ==================================================
        # CONVERT TARGET TO READABLE RESULT
        # ==================================================

        if prediction == 1:

            result = "High Risk"

            message = (
                "The model predicted class 1 for this input."
            )

            result_type = "high"

        else:

            result = "Low Risk"

            message = (
                "The model predicted class 0 for this input."
            )

            result_type = "low"


        # ==================================================
        # RETURN RESULT
        # ==================================================

        return render_template(

            "index.html",

            prediction=result,

            message=message,

            result_type=result_type

        )


    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return render_template(

            "index.html",

            error=f"Prediction error: {str(e)}"

        )


# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )