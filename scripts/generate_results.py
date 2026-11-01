import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "diabetes_clustered.csv"
RESULTS_DIR = BASE_DIR / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=10, solver="lbfgs", random_state=42, max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 0]  # Probability of High Risk (class 0)

    # 1. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_high = precision_score(y_test, y_pred, pos_label=0)
    rec_high = recall_score(y_test, y_pred, pos_label=0)
    f1_high = f1_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test == 0, y_prob)

    report_dict = classification_report(y_test, y_pred, target_names=["High Risk (0)", "Low Risk (1)"], output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_high_risk": round(float(prec_high), 4),
        "recall_high_risk": round(float(rec_high), 4),
        "f1_high_risk": round(float(f1_high), 4),
        "roc_auc_score": round(float(roc_auc), 4),
        "total_test_samples": len(y_test),
        "train_samples": len(X_train),
        "classification_report": report_dict,
    }

    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved: Accuracy={acc:.4f}, Recall (High Risk)={rec_high:.4f}, F1={f1_high:.4f}")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax)

    labels = ["High Risk (0)", "Low Risk (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Diabetes Risk Classifier - Confusion Matrix\n(Test Set N=154)", fontsize=13, pad=12, fontweight="bold")

    matrix_labels = [["True High Risk (TP)", "False Low Risk (FN)"], ["False High Risk (FP)", "True Low Risk (TN)"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i, f"{matrix_labels[i][j]}\n{cm[i, j]}",
                ha="center", va="center", color=color, fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close(fig)

    # 3. Feature Importance (Coefficients)
    model = pipeline.named_steps["model"]
    # Class 0 = High Risk, Class 1 = Low Risk. Negative coef indicates higher feature values push toward class 0.
    coefs = -model.coef_[0]  # flip sign so positive coef = higher risk
    sorted_idx = np.argsort(coefs)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.barh(range(len(FEATURES)), coefs[sorted_idx], color="#1f77b4", edgecolor="#0e4b75")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURES[i] for i in sorted_idx], fontsize=11)
    ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
    ax.set_xlabel("Standardized Impact on High Risk Score (Log-Odds)", fontsize=11)
    ax.set_title("Feature Importance / Impact on High Diabetes Risk", fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.05 if width >= 0 else -0.05
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                va="center", ha=ha, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png")
    plt.close(fig)

    # 4. Sample Predictions
    sample_df = X_test.head(5).copy()
    sample_preds = pipeline.predict(sample_df)
    sample_probs = pipeline.predict_proba(sample_df)
    samples = []
    for idx, (original_idx, row) in enumerate(sample_df.iterrows()):
        samples.append({
            "features": row.to_dict(),
            "true_label": int(y_test.loc[original_idx]),
            "true_risk": "high" if y_test.loc[original_idx] == 0 else "low",
            "predicted_label": int(sample_preds[idx]),
            "predicted_risk": "high" if sample_preds[idx] == 0 else "low",
            "high_risk_probability": round(float(sample_probs[idx][0]), 4),
            "low_risk_probability": round(float(sample_probs[idx][1]), 4),
        })

    with open(RESULTS_DIR / "sample_predictions.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("All results and plots generated successfully in docs/results/!")

if __name__ == "__main__":
    main()
