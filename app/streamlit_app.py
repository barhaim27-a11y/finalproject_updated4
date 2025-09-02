import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy, shap, io
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
)
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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.title("⚙️ Settings")

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, key="theme_choice")
language_choice = st.sidebar.selectbox("Language", ["English", "עברית", "Français"], index=0, key="lang_choice")
text_size = st.sidebar.select_slider("Text Size", options=["Small", "Medium", "Large"], value="Medium", key="text_size")
layout_density = st.sidebar.radio("Layout Density", ["Comfortable", "Compact"], index=0, key="layout_density")
show_advanced_eda = st.sidebar.checkbox("Show Advanced EDA Visualizations", value=True, key="eda_toggle")

threshold_global = st.sidebar.slider("Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01, key="global_threshold")

def apply_custom_style():
    css = ""
    if theme_choice == "Dark":
        css += "body, .stApp { background-color:#111 !important; color:#eee !important; }"
    if text_size == "Small":
        css += "body, .stApp { font-size:13px !important; }"
    elif text_size == "Medium":
        css += "body, .stApp { font-size:16px !important; }"
    elif text_size == "Large":
        css += "body, .stApp { font-size:19px !important; }"
    if layout_density == "Compact":
        css += ".block-container { padding-top:0rem; padding-bottom:0rem; }"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_style()

# ==============================
# Helpers
# ==============================
def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]
    return X

def safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception:
        return model.predict(align_features(model, X))

def safe_predict_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        return model.predict_proba(align_features(model, X))

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
    with open("assets/metrics.json", "r") as f:
        metrics = json.load(f)
    return best_model, metrics

if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

# ==============================
# Tabs
# ==============================
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_explain, tab_history, tab_about = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "🧠 Explainability",
    "📜 Model History",
    "ℹ️ About"
])

# ==============================
# Tab 1: Data & EDA
# ==============================
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.describe().T)

# ==============================
# Tab 2: Dashboard
# ==============================
with tab_dash:
    st.header("📈 Compare Models")
    st.info("Interactive dashboard for training multiple models with parameters.")

# ==============================
# Tab 3: Models
# ==============================
with tab2:
    st.header("🤖 Model Training & Comparison")
    st.dataframe(pd.DataFrame(metrics).T)

# ==============================
# Tab 4: Prediction
# ==============================
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)

    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])

    if option == "Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])

        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0, 1]
            pred = int(prob >= threshold)
            st.subheader("🧾 Prediction Result")

            if pred == 1:
                st.error(f"🔴 Parkinson’s Detected ({prob*100:.1f}%)")
            else:
                st.success(f"🟢 Healthy ({prob*100:.1f}%)")

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text": "Probability of Parkinson’s"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if pred==1 else "green"}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            # PDF Report
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Parkinson’s Prediction Report")
            c.drawString(100, 720, f"Prediction: {'Parkinson’s Detected' if pred==1 else 'Healthy'}")
            c.drawString(100, 700, f"Probability: {prob*100:.1f}%")
            c.drawString(100, 680, f"Threshold: {threshold:.2f}")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("📥 Download Prediction Report (PDF)", pdf_buffer, "prediction_report.pdf", "application/pdf")

# ==============================
# Tab 5: Test Evaluation
# ==============================
with tab5:
    st.header("🧪 Model Evaluation on External Test Set")
    st.info("Upload a test CSV with 'status' column.")

# ==============================
# Tab 6: Train New Model
# ==============================
with tab4:
    st.header("⚡ Train New Model")
    st.info("Upload new dataset, choose models and compare results.")

# ==============================
# Tab 7: Explainability
# ==============================
with tab_explain:
    st.header("🧠 Explainability")
    st.info("Feature importance and SHAP plots will be shown here.")

# ==============================
# Tab 8: Model History
# ==============================
with tab_history:
    st.header("📜 Model History")
    history_file = "assets/model_history.csv"
    if os.path.exists(history_file):
        hist = pd.read_csv(history_file)
        st.dataframe(hist)
        if st.button("⏪ Rollback to Previous Model"):
            st.warning("Rollback executed (demo).")
    else:
        st.info("No history found yet.")

# ==============================
# Tab 9: About
# ==============================
with tab_about:
    st.header("ℹ️ About")
    st.markdown("""
    **Parkinson’s ML App**  
    Version: 1.0  
    Developed for demonstration.  
    Features: EDA, Dashboard, Prediction, Training, Explainability, History.
    """)
