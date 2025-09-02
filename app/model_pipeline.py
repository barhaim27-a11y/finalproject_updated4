# app/model_pipeline.py
import os, json, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Paths
DATA_PATH = os.path.join("data", "parkinsons.csv")
MODELS_DIR = "models"
ASSETS_DIR = "assets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ======================
# Load data
# ======================
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# Define models
# ======================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ]),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "XGBoost": xgb.XGBClassifier(
        eval_metric="logloss", use_label_encoder=False, random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(
        verbose=0, random_state=42
    ),
    "NeuralNet": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                              max_iter=500, random_state=42))
    ])
}

# ======================
# Train + Evaluate
# ======================
metrics = {}
full_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics[name] = roc_auc
    full_results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }

# ======================
# Save metrics
# ======================
with open(os.path.join(ASSETS_DIR, "metrics.json"), "w") as f:
    json.dump(full_results, f, indent=4)

# ======================
# Select best model
# ======================
best_name = max(metrics, key=metrics.get)
best_model = models[best_name]
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
print(f"✅ Best model: {best_name} (ROC-AUC={metrics[best_name]:.3f})")

# ======================
# Save plots
# ======================
# Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test))
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Healthy","Parkinson’s"])
ax.set_yticklabels(["Healthy","Parkinson’s"])
plt.colorbar(im)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"))
plt.close(fig)

# ROC Curve
y_proba = best_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"{best_name} (AUC={metrics[best_name]:.2f})")
ax.plot([0,1],[0,1],'k--')
ax.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(ASSETS_DIR, "roc_curve.png"))
plt.close(fig)

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(rec, prec, label=best_name)
ax.legend()
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(ASSETS_DIR, "pr_curve.png"))
plt.close(fig)

# SHAP Summary
try:
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
    fig = shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"))
    plt.close()
except Exception as e:
    print("⚠️ Could not compute SHAP values:", e)
