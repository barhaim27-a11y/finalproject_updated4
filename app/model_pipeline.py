import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

# ==============================
# Load dataset
# ==============================
df = pd.read_csv("data/parkinsons.csv")
X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Load config (from Streamlit)
# ==============================
config_path = "assets/config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {
        "models": ["LogisticRegression","RandomForest","XGBoost"],
        "params": {"rf_trees": 200, "xgb_lr": 0.1}
    }

chosen_models = config.get("models", [])
params = config.get("params", {})

# ==============================
# Define models
# ==============================
all_models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=params.get("rf_trees", 200), random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, kernel="rbf"))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "XGBoost": xgb.XGBClassifier(
        eval_metric="logloss", random_state=42,
        learning_rate=params.get("xgb_lr", 0.1)
    ),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "NeuralNet": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))
    ])
}

# Filter models by chosen list
models = {name: m for name, m in all_models.items() if name in chosen_models}

# ==============================
# Train & Evaluate
# ==============================
metrics = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

# ==============================
# Save Results
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

with open("assets/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save best model
best_name = max(metrics, key=lambda m: metrics[m]["roc_auc"])
joblib.dump(models[best_name], "models/best_model.joblib")

print(f"✅ Best model: {best_name} (ROC-AUC={metrics[best_name]['roc_auc']:.3f})")
