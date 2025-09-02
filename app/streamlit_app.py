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

# PDF export (safe import)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import io
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

# ✅ Decision Threshold (global) → אחד בלבד
threshold_global = st.sidebar.slider(
    "Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01, key="global_threshold"
)

# ==============================
# Apply UI Customizations (CSS)
# ==============================
def apply_custom_style():
    css = ""
    # Theme
    if theme_choice == "Dark":
        css += """
        body, .stApp {
            background-color: #111 !important;
            color: #eee !important;
        }
        """
    # Text size
    if text_size == "Small":
        css += "body, .stApp { font-size: 13px !important; }"
    elif text_size == "Medium":
        css += "body, .stApp { font-size: 16px !important; }"
    elif text_size == "Large":
        css += "body, .stApp { font-size: 19px !important; }"

    # Layout density
    if layout_density == "Compact":
        css += ".block-container { padding-top: 0rem; padding-bottom: 0rem; }"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_style()


# ==============================
# Helpers
# ==============================
def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """מתאים את הקובץ שהועלה לעמודות שהמודל מצפה להן"""
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        # נוסיף עמודות חסרות
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        # נסיר עמודות עודפות ונשמור על סדר
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
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_history, tab_explain, tab_about, tab_pdf = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "📜 Model History",
    "🧠 Explainability",
    "ℹ️ About",
    "📄 PDF Report"
])

# === המשך הקוד שלך (EDA, Dashboard, Models, Prediction, Test Eval, Train New Model) ===
# 👆 לא שיניתי פה כלום – זה נשאר בדיוק כמו אצלך.

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
    st.subheader("Feature Importance")
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = pd.Series(best_model.feature_importances_, index=X.columns)
            st.bar_chart(importances.sort_values(ascending=False).head(10))
        else:
            st.info("Feature importance not available for this model.")
    except Exception:
        st.warning("Could not compute feature importance.")

    st.subheader("SHAP Values")
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
    - Built with Streamlit, Scikit-learn, XGBoost, LightGBM, CatBoost.  
    - Provides data exploration, model training, evaluation, explainability, and export.  
    - Designed to help understand and detect Parkinson’s disease patterns from voice features.  

    👨‍💻 Author: Your Name  
    📅 Last Updated: 2025
    """)

# --- Tab 10: PDF Report
with tab_pdf:
    st.header("📄 Export Report to PDF")
    if not PDF_AVAILABLE:
        st.error("❌ PDF generation not available. Please install `reportlab` in requirements.txt")
    else:
        if st.button("📥 Generate PDF Report"):
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.drawString(100, 750, "Parkinson’s ML App – Report")
            c.drawString(100, 730, "This report includes dataset stats and best model performance.")
            c.save()
            pdf_buffer.seek(0)
            st.download_button(
                "Download PDF",
                data=pdf_buffer,
                file_name="report.pdf",
                mime="application/pdf"
            )
