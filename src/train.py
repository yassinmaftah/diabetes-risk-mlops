import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_clustered.csv"
FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
MODEL_NAME = "diabetes-risk-classifier"



def train_and_save(data_path=DATA_PATH):
    
    tracking_url = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_url)
    
    mlflow.set_experiment("diabetes-classification")
    
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df["Cluster"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=10, solver="lbfgs", random_state=42, max_iter=1000)),
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    with mlflow.start_run(run_name="train-logistic-pipeline"):
        mlflow.log_params({"model": "LogisticRegression", "C": 10, "solver": "lbfgs",
                           "features": ", ".join(FEATURES)})
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision_high_risk", precision_score(y_test, y_pred, pos_label=0))
        mlflow.log_metric("recall_high_risk", recall_score(y_test, y_pred, pos_label=0))
        mlflow.log_metric("f1_high_risk", f1_score(y_test, y_pred, pos_label=0))
        
        info = mlflow.sklearn.log_model(pipeline, name="model",
                                        registered_model_name=MODEL_NAME)
        
        version = info.registered_model_version
        
    MlflowClient().set_registered_model_alias(MODEL_NAME, "Production", version)
    return version

if __name__ == "__main__":
    version = train_and_save()
    print(f"Model version {version} registered and set to Production")
    
