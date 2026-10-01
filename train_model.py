import os
import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


# ==========================================
# APP
# ==========================================

app = FastAPI()


# ==========================================
# PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================
# STATIC FILES
# ==========================================

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ==========================================
# MODEL
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_pipeline.pkl"
)

model = joblib.load(
    MODEL_PATH
)


# ==========================================
# TEMPLATES
# ==========================================

templates = Jinja2Templates(
    directory=os.path.join(
        BASE_DIR,
        "templates"
    )
)


# ==========================================
# HOME
# ==========================================

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "risk_percentage": None
        }
    )


# ==========================================
# PREDICTION
# ==========================================

@app.post(
    "/predict",
    response_class=HTMLResponse
)
async def predict(request: Request):

    form = await request.form()


    # --------------------------------------
    # GET FORM VALUES
    # --------------------------------------

    data = {
        "age": float(form["age"]),
        "sex": float(form["sex"]),
        "cp": float(form["cp"]),
        "trestbps": float(form["trestbps"]),
        "chol": float(form["chol"]),
        "fbs": float(form["fbs"]),
        "restecg": float(form["restecg"]),
        "thalach": float(form["thalach"]),
        "exang": float(form["exang"]),
        "oldpeak": float(form["oldpeak"]),
        "slope": float(form["slope"]),
        "ca": float(form["ca"]),
        "thal": float(form["thal"])
    }


    # --------------------------------------
    # DATAFRAME
    # --------------------------------------

    input_data = pd.DataFrame(
        [data]
    )


    # --------------------------------------
    # PREDICTION
    # --------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------
    # PROBABILITY
    # --------------------------------------

    probabilities = model.predict_proba(
        input_data
    )[0]


    # Find probability of positive class
    classes = model.classes_

    if 1 in classes:

        positive_index = list(
            classes
        ).index(1)

        risk_percentage = (
            probabilities[positive_index] * 100
        )

    else:

        risk_percentage = (
            max(probabilities) * 100
        )


    risk_percentage = round(
        risk_percentage,
        2
    )


    # --------------------------------------
    # RESULT
    # --------------------------------------

    if prediction == 1:

        result = "Higher Predicted Risk"

    else:

        result = "Lower Predicted Risk"


    # --------------------------------------
    # RETURN DASHBOARD
    # --------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": result,
            "risk_percentage": risk_percentage
        }
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
def health_check():

    return {
        "status": "running",
        "message": "Heart Disease Prediction API"
    }

