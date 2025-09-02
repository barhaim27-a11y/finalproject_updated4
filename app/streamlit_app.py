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
from sklearn.model_selection import train_test_split, learning_curve
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
# Helpers (עודכן עם align_features)
# ==============================
def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """מתאים את הדאטה לדרישות המודל"""
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        # הוספת עמודות חסרות
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        # סידור לפי סדר העמודות שהמודל מצפה
        X = X[expected_cols]
    return X

def safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception:
        X = align_features(model, X)
        return model.predict(X)

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        X = align_features(model, X)
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
# Tabs
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

# --- Tab 2: Dashboard (קיצור – כמו בגירסה הקודמת שלך)
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")
    # ... כאן נשאר הקוד של ההשוואה בין המודלים עם ROC/PR ...

# --- Tab 3: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    df_metrics = df_metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df_metrics.insert(0, "Rank", df_metrics.index + 1)
    best_name = df_metrics.iloc[0]["Model"]
    df_metrics.loc[0, "Model"] = f"🏆 {best_name}"

    st.subheader("📊 Model Ranking")
    st.dataframe(df_metrics)

    st.subheader("🏆 Best Model Results")
    best_row = df_metrics.iloc[0]
    st.table(pd.DataFrame(best_row).T)

    # Confusion Matrix
    y_pred = safe_predict(best_model, X)
    y_pred_prob = safe_predict_proba(best_model, X)[:,1]
    cm = confusion_matrix(y, y_pred)
    fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # Learning Curve
    st.subheader("Learning Curve – Best Model")
    train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1), mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(test_scores, axis=1), mode="lines+markers", name="Validation"))
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Prediction (גרסה מעוצבת)
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)

    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])

    if option=="Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])

        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
            pred = int(prob >= threshold)

            st.subheader("🧾 Prediction Result")
            if pred == 1:
                st.markdown(f"""
                <div style="padding:15px; background-color:#ffe6e6; border-radius:10px; text-align:center">
                    <h2 style="color:#cc0000">🔴 Parkinson’s Detected</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:15px; background-color:#e6ffe6; border-radius:10px; text-align:center">
                    <h2 style="color:#009900">🟢 Healthy</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob*100,
                title = {"text": "Probability of Parkinson’s"},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if pred==1 else "green"},
                    "steps": [
                        {"range": [0,30], "color":"#e6ffe6"},
                        {"range": [30,70], "color":"#fff5e6"},
                        {"range": [70,100], "color":"#ffe6e6"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    else:
        file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
        if file:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(new_df.head())

            probs = safe_predict_proba(best_model, new_df)[:,1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = (probs*100).round(1)
            new_df["Prediction"] = preds

            st.subheader("📊 Prediction Summary")
            summary = pd.Series(preds).value_counts().rename({0:"Healthy 🟢",1:"Parkinson’s 🔴"})
            st.table(summary)

            st.subheader("Detailed Results")
            st.dataframe(new_df.head(20))

            st.download_button("📥 Download Predictions (CSV)", new_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            new_df.to_excel("predictions.xlsx", index=False)
            with open("predictions.xlsx","rb") as f:
                st.download_button("📥 Download Predictions (Excel)", f, "predictions.xlsx", "application/vnd.ms-excel")

# --- Tab 5: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    model_choices = st.multiselect("Select Models", ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"], default=["RandomForest","XGBoost"])
    rf_trees = st.slider("RandomForest Trees", 50, 500, 200, 50)
    xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.5, 0.1, 0.01)

    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())

        if st.button("Retrain Models"):
            new_df.to_csv("data/new_train.csv", index=False)
            config = {"models": model_choices, "params": {"rf_trees": rf_trees, "xgb_lr": xgb_lr}}
            os.makedirs("assets", exist_ok=True)
            with open("assets/config.json","w") as f:
                json.dump(config, f, indent=4)
            runpy.run_path("app/model_pipeline.py")

            with open("assets/metrics.json","r") as f:
                new_metrics = json.load(f)
            comp_df = pd.DataFrame(new_metrics).T.reset_index().rename(columns={"index":"Model"})
            st.subheader("📊 New Training Results")
            st.dataframe(comp_df)
            st.success("✅ Models retrained! See results above.")
