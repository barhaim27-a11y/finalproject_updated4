import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy, shap
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

st.sidebar.title("⚙️ Settings")
threshold_global = st.sidebar.slider("Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01)

# ==============================
# Helpers
# ==============================
def safe_predict(model, X):
    try:
        return model.predict(X)
    except ValueError:
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        return model.predict(X)

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except ValueError:
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
        return model.predict_proba(X)

# ==============================
# Load dataset
# ==============================
DATA_PATH = "data/parkinsons.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop("status", axis=1)
y = df["status"]

# ==============================
# Load model + metrics
# ==============================
def load_model_and_metrics():
    if not os.path.exists("models/best_model.joblib") or not os.path.exists("assets/metrics.json"):
        runpy.run_path("app/model_pipeline.py")
    best_model = joblib.load("models/best_model.joblib")
    with open("assets/metrics.json","r") as f:
        metrics = json.load(f)
    return best_model, metrics

if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

# ==============================
# Tabs (הוספנו Dashboard כטאב 2)
# ==============================
tab1, tab_dash, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "⚡ Train New Model"
])

# --- Tab 1: Data & EDA
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info & Statistics")
    st.write(f"🔹 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning("Missing Values detected:")
        st.dataframe(missing[missing > 0])
    else:
        st.success("No missing values ✅")
    st.dataframe(df.describe().T)
    st.table(y.value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

    st.write("🔹 Top Features Correlated with Target")
    corr_target = df.corr()["status"].abs().sort_values(ascending=False)[1:6]
    st.table(corr_target)

    st.subheader("Exploratory Plots")
    eda_dir = "eda"
    eda_plots = {
        "Target Distribution (Count & Pie)": "target_distribution_combo.png",
        "Correlation Heatmap": "corr_heatmap.png",
        "Pairplot of Top Features": "pairplot_top_features.png",
        "Histograms & Violin Plots": "distributions_violin.png",
        "PCA Projection": "pca.png",
        "t-SNE Projection": "tsne.png"
    }
    for title, filename in eda_plots.items():
        path = os.path.join(eda_dir, filename)
        if os.path.exists(path):
            with st.expander(title, expanded=False):
                st.image(path, use_column_width=True)

# --- Tab 2: Dashboard (New)
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")

    # בחירת מודלים
    model_options = ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"]
    chosen_models = st.multiselect("בחר מודלים להשוואה", model_options, default=["RandomForest","XGBoost"])

    # Hyperparameters
    st.subheader("⚙️ Hyperparameters")
    params = {}
    if "RandomForest" in chosen_models:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5)
        }
    if "XGBoost" in chosen_models:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50)
        }
    if "SVM" in chosen_models:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1)
        }

    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        trained_models = {}
        metrics_comp = {}

        for m in chosen_models:
            if m == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=params[m]["n_estimators"], 
                    max_depth=params[m]["max_depth"], 
                    random_state=42
                )
            elif m == "XGBoost":
                model = xgb.XGBClassifier(
                    eval_metric="logloss", 
                    n_estimators=params[m]["n_estimators"], 
                    learning_rate=params[m]["learning_rate"], 
                    random_state=42
                )
            elif m == "SVM":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(C=params[m]["C"], probability=True, kernel="rbf"))
                ])
            elif m == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
            elif m == "KNN":
                model = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(random_state=42)
            elif m == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            elif m == "NeuralNet":
                model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))])
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_val = roc_auc_score(y_test, y_proba)

            trained_models[m] = model
            metrics_comp[m] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc_val}

        # טבלת מדדים
        st.subheader("📊 Metrics Comparison")
        df_comp = pd.DataFrame(metrics_comp).T
        st.dataframe(df_comp.style.highlight_max(axis=0))

        # ROC curves
        st.subheader("ROC Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
        st.plotly_chart(fig, use_container_width=True)

        # PR curves
        st.subheader("Precision-Recall Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=m))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Models (מקורי + שדרוגים נשאר פה)
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(df_metrics)

    kpi_cols = st.columns(5)
    best_row = df_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    for i, k in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
        if k in best_row:
            kpi_cols[i].metric(k.capitalize(), f"{best_row[k]:.3f}")
