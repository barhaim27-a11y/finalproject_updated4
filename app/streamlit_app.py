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

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.title("⚙️ Settings")

# ✅ Theme selector
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, key="theme_choice")

# ✅ Language selector
language_choice = st.sidebar.selectbox(
    "Language", ["English", "עברית", "العربية", "Français"], index=1, key="lang_choice"
)

# ✅ Text size
text_size = st.sidebar.select_slider(
    "Text Size", options=["Small", "Medium", "Large"], value="Medium", key="text_size"
)

# ✅ Layout density
layout_density = st.sidebar.radio(
    "Layout Density", ["Comfortable", "Compact"], index=0, key="layout_density"
)

# ✅ Toggle advanced EDA
show_advanced_eda = st.sidebar.checkbox(
    "Show Advanced EDA Visualizations", value=True, key="eda_toggle"
)

# ✅ Decision Threshold (global)
threshold_global = st.sidebar.slider(
    "Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01, key="global_threshold"
)

# ==============================
# Apply UI Customizations (CSS)
# ==============================
def apply_custom_style():
    css = ""
    if theme_choice == "Dark":
        css += """
        body, .stApp {
            background-color: #111 !important;
            color: #eee !important;
        }
        """
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
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_explain, tab_about = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "🧠 Explainability",
    "ℹ️ About"
])

# --- Tab 1: Data & EDA
with tab1:
    ...
    # (התוכן שלך נשאר בדיוק כמו בקוד ששלחת)

# --- Tab 2: Dashboard
with tab_dash:
    ...
    # (התוכן שלך נשאר בדיוק כמו בקוד ששלחת)

# --- Tab 3: Models
with tab2:
    ...
    # (התוכן שלך נשאר בדיוק כמו בקוד ששלחת)

# --- Tab 4: Prediction
with tab3:
    ...
    # בסוף חיזוי יחיד הוסף PDF Report
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Parkinson’s Prediction Report")
            c.drawString(100, 720, f"Prediction: {'Parkinson’s Detected' if pred else 'Healthy'}")
            c.drawString(100, 700, f"Probability: {prob*100:.1f}%")
            c.save()
            pdf_buffer.seek(0)
            st.download_button("📥 Download PDF Report", pdf_buffer, "prediction_report.pdf", "application/pdf")

# --- Tab 5: Test Evaluation
with tab5:
    ...
    # (התוכן שלך נשאר בדיוק כמו בקוד ששלחת)

# --- Tab 6: Train New Model
with tab4:
    ...
        if st.button("🚀 Retrain Models"):
            with st.spinner("⏳ Training models..."):
                # (הקוד שלך נשאר – עטפתי בספינר)
                ...
            st.success("✅ Training complete!")

            # --- Model History
            history_path = "assets/model_history.csv"
            df_comp.to_csv(history_path, mode="a", header=not os.path.exists(history_path), index=True)
            if os.path.exists(history_path):
                hist_df = pd.read_csv(history_path)
                st.subheader("📜 Model History (Last 10)")
                st.dataframe(hist_df.tail(10))
                st.download_button("📥 Download History", hist_df.to_csv(index=False).encode("utf-8"), "model_history.csv", "text/csv")

            # --- Rollback option
            rollback_path = "models/best_model_backup.joblib"
            if os.path.exists(rollback_path):
                if st.button("↩️ Rollback to Previous Model"):
                    joblib.copy(rollback_path, "models/best_model.joblib")
                    st.success("✅ Rolled back to previous best model.")

# --- Tab 7: Explainability
with tab_explain:
    st.header("🧠 Model Explainability")
    try:
        explainer = shap.Explainer(best_model, X)
        shap_values = explainer(X)
        st.subheader("Feature Importance (SHAP)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP not available: {e}")

# --- Tab 8: About
with tab_about:
    st.header("ℹ️ About This App")
    st.markdown("""
    **Parkinson’s ML App**  
    - Predict Parkinson’s disease using ML models  
    - Compare and retrain models  
    - Explain predictions with SHAP  
    - Export results to PDF/Excel  
    """)
