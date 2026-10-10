import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]

app = FastAPI(title="Diabetes Risk API")
model = None


class Patient(BaseModel):
    Glucose: float = Field(gt=0, lt=400)
    BloodPressure: float = Field(gt=0, lt=250)
    SkinThickness: float = Field(gt=0, lt=100)
    Insulin: float = Field(gt=0, lt=1000)
    BMI: float = Field(gt=0, lt=100)
    DiabetesPedigreeFunction: float = Field(gt=0, lt=5)
    Age: float = Field(gt=0, lt=120)


def get_model():
    global model
    if model is None:
        mlflow.set_tracking_uri(TRACKING_URI)
        model = mlflow.sklearn.load_model("models:/diabetes-risk-classifier@Production")
    return model


@app.get("/health")
def health():
    return {"status": "ok for test"}


@app.post("/predict")
def predict(patient: Patient):
    try:
        data = pd.DataFrame([{
                "Glucose": patient.Glucose,
                "BloodPressure": patient.BloodPressure,
                "SkinThickness": patient.SkinThickness,
                "Insulin": patient.Insulin,
                "BMI": patient.BMI,
                "DiabetesPedigreeFunction": patient.DiabetesPedigreeFunction,
                "Age": patient.Age
            }])[FEATURES]
        prediction = int(get_model().predict(data)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    risk = "high" if prediction == 0 else "low"
    return {"prediction": prediction, "risk_level": risk}