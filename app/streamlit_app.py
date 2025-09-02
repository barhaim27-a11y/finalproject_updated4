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
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import PartialDependenceDisplay

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# Sidebar – Extra Settings
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
# Tabs (same structure as before)
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA", 
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

    # גרפים (מהתיקייה EDA)
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

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")

    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(df_metrics)

    # KPI Cards
    kpi_cols = st.columns(5)
    best_row = df_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    for i, k in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
        if k in best_row:
            kpi_cols[i].metric(k.capitalize(), f"{best_row[k]:.3f}")

    # Confusion Matrix (Plotly)
    st.subheader("Confusion Matrix (Best Model)")
    y_pred = safe_predict(best_model, X)
    cm = confusion_matrix(y, y_pred)
    fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve (Plotly)
    st.subheader("ROC Curve (Best Model)")
    y_pred_prob = safe_predict_proba(best_model, X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={best_row['roc_auc']:.2f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
    st.plotly_chart(fig, use_container_width=True)

    # PR Curve
    st.subheader("Precision-Recall Curve")
    prec, rec, _ = precision_recall_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve"))
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        st.subheader("Feature Importance")
        importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig = px.bar(importances, orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    # PDP
    try:
        st.subheader("Partial Dependence Plot")
        fig, ax = plt.subplots()
        PartialDependenceDisplay.from_estimator(best_model, X, [0], ax=ax)
        st.pyplot(fig)
    except Exception:
        st.info("PDP not available")

    # SHAP
    shap_path = os.path.join("assets","shap_summary.png")
    if os.path.exists(shap_path):
        st.subheader("Explainability (SHAP)")
        st.image(shap_path, caption="SHAP Summary")
    else:
        try:
            explainer = shap.Explainer(best_model, X)
            shap_values = explainer(X.iloc[:50])
            st.pyplot(shap.plots.force(shap_values[0], matplotlib=True, show=False))
        except Exception:
            st.warning("SHAP not available")

# --- Tab 3: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)

    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])
    if option=="Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
            st.progress(prob)
            st.write(f"Probability: {prob:.2f}")
    else:
        file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
        if file:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)

            st.write("Preview:")
            st.dataframe(new_df.head())

            probs = safe_predict_proba(best_model, new_df)[:,1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = probs
            new_df["Prediction"] = preds

            st.write("Summary of Predictions")
            st.table(new_df["Prediction"].value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

            # Confusion Matrix dynamic
            cm = confusion_matrix(preds, new_df["Prediction"])
            fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Oranges")
            st.plotly_chart(fig)

            # Downloads
            st.download_button("📥 Download Predictions (CSV)", new_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    model_choices = st.multiselect(
        "Select Models", 
        ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"],
        default=["RandomForest","XGBoost"]
    )
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
