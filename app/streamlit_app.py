import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy, shap, io
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

# PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.title("⚙️ Settings")

theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, key="theme_choice")
language_choice = st.sidebar.selectbox("Language", ["English", "עברית", "العربية", "Français"], index=1, key="lang_choice")
text_size = st.sidebar.select_slider("Text Size", options=["Small", "Medium", "Large"], value="Medium", key="text_size")
layout_density = st.sidebar.radio("Layout Density", ["Comfortable", "Compact"], index=0, key="layout_density")
show_advanced_eda = st.sidebar.checkbox("Show Advanced EDA Visualizations", value=True, key="eda_toggle")
threshold_global = st.sidebar.slider("Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01, key="global_threshold")

# ==============================
# Apply UI Customizations (CSS)
# ==============================
def apply_custom_style():
    css = ""
    if theme_choice == "Dark":
        css += "body, .stApp { background-color: #111 !important; color: #eee !important; }"
    if text_size == "Small":
        css += "body, .stApp { font-size: 13px !important; }"
    elif text_size == "Medium":
        css += "body, .stApp { font-size: 16px !important; }"
    elif text_size == "Large":
        css += "body, .stApp { font-size: 19px !important; }"
    if layout_density == "Compact":
        css += ".block-container { padding-top: 0rem; padding-bottom: 0rem; }"
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
    try: return model.predict(X)
    except Exception: return model.predict(align_features(model, X))

def safe_predict_proba(model, X):
    try: return model.predict_proba(X)
    except Exception: return model.predict_proba(align_features(model, X))

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
    with open("assets/metrics.json","r") as f: metrics = json.load(f)
    return best_model, metrics

if "best_model" not in st.session_state or "metrics" not in st.session_state:
    st.session_state.best_model, st.session_state.metrics = load_model_and_metrics()

best_model = st.session_state.best_model
metrics = st.session_state.metrics

# ==============================
# Tabs
# ==============================
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_history, tab_explain, tab_about, tab_pdf = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "📜 History",
    "🧠 Explainability",
    "ℹ️ About",
    "📄 PDF Report"
])

# --- Tab 1: Data & EDA
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    # 🔹 שאר הקוד שלך ל־EDA נשאר כמו שהיה

# --- Tab 2: Dashboard
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")
    # 🔹 כל הקוד שלך מהדאשבורד כאן

# --- Tab 3: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    # 🔹 כל הקוד שלך למודלים, Confusion Matrix, ROC וכו'

# --- Tab 4: Prediction
with tab3:
    st.header("🔮 Prediction")
    # 🔹 כל הקוד שלך לפרדיקציה (ידני/קובץ)

# --- Tab 5: Test Evaluation
with tab5:
    st.header("🧪 Model Evaluation on External Test Set")
    # 🔹 כל הקוד שלך ל־Test Evaluation

# --- Tab 6: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    # 🔹 כל הקוד שלך לאימון מחדש והשוואה מול Best Model

# --- Tab 7: Model History
with tab_history:
    st.header("📜 Model History")
    history_path = "assets/model_history.csv"
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        st.dataframe(hist_df)
        if st.button("🔙 Rollback to Previous Best"):
            last_model_path = hist_df.iloc[-2]["model_path"]
            if os.path.exists(last_model_path):
                best_model = joblib.load(last_model_path)
                st.success("✅ Rolled back to previous model!")
    else:
        st.info("No history available yet.")

# --- Tab 8: Explainability
with tab_explain:
    st.header("🧠 Explainability")
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=X.columns)
            st.bar_chart(importances.sort_values(ascending=False).head(10))
        else:
            st.info("Feature importance not available for this model.")
    except Exception:
        st.warning("Could not compute feature importance.")

    try:
        explainer = shap.Explainer(best_model, X)
        shap_values = explainer(X)
        st.pyplot(shap.summary_plot(shap_values, X, show=False))
    except Exception:
        st.info("SHAP not available for this model.")

# --- Tab 9: About
with tab_about:
    st.header("ℹ️ About this App")
    st.markdown("""
    🧠 **Parkinson’s ML App**  
    - Built with Streamlit, Scikit-learn, XGBoost, LightGBM, CatBoost  
    - Provides EDA, model training, evaluation, explainability, export  
    👨‍💻 Author: Your Name  
    📅 Last Updated: 2025
    """)

# --- Tab 10: PDF Report
with tab_pdf:
    st.header("📄 Export Report to PDF")
    if not PDF_AVAILABLE:
        st.error("❌ PDF generation not available. Please install `reportlab`")
    else:
        if st.button("📥 Generate PDF Report"):
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Parkinson’s ML App – Report")
            c.drawString(100, 730, "This report includes dataset stats and best model performance.")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("Download PDF", data=pdf_buffer, file_name="report.pdf", mime="application/pdf")
