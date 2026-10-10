from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import hashlib
import platform

import numpy as np
import pandas as pd
import sklearn

import mlflow
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI",
                         f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
# sqlite:///C:/Users/yassi/Desktop/diabetes-risk-mlops/mlflow.db

def log_clustering_run(kmeans, scaler, features, silhouette, model_path, scaler_path,
                    run_name="kmeans-k2"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"diabetes-clustering")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("k", kmeans.n_clusters)
        mlflow.log_param("features", ", ".join(features))

        for name, mean, std in zip(features, scaler.mean_, scaler.scale_):
            mlflow.log_param(f"scaler_mean_{name}", round(mean, 4))
            mlflow.log_param(f"scaler_std_{name}", round(std, 4))

        mlflow.log_metric("inertia", kmeans.inertia_)
        mlflow.log_metric("silhouette_score", silhouette)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        
        
        
def log_classification_run(pipeline, X_test, y_test, params, features, run_name,
                           model_path):
    y_pred = pipeline.predict(X_test)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"diabetes-classification")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("features", ", ".join(features))

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision_high_risk", precision_score(y_test, y_pred, pos_label=0))
        mlflow.log_metric("recall_high_risk", recall_score(y_test, y_pred, pos_label=0))
        mlflow.log_metric("f1_high_risk", f1_score(y_test, y_pred, pos_label=0))

        mlflow.log_artifact(model_path)
        
        image_path = PROJECT_ROOT / "models" / f"confusion_matrix_{run_name}.png"
        save_confusion_matrix_image(y_test, y_pred, image_path, run_name)
        mlflow.log_artifact(str(image_path))
        
        log_library_versions()        
    
def save_confusion_matrix_image(y_test, y_pred, path, title):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    names = [["TP", "FN"], ["FP", "TN"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}", ha="center", va="center", fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["High Risk (0)", "Low Risk (1)"])
    ax.set_yticklabels(["High Risk (0)", "Low Risk (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    
    
def log_library_versions():
    mlflow.log_params({
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "mlflow_version": mlflow.__version__,
    })