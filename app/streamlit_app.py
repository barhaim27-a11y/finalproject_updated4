import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy
import matplotlib.pyplot as plt
import shap

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.inspection import PartialDependenceDisplay

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML Dashboard", page_icon="🧠", layout="wide")

# Sidebar – Settings
st.sidebar.title("⚙️ Settings")
lang = st.sidebar.radio("🌐 Language", ["English","עברית"], index=0)
theme = st.sidebar.radio("🎨 Theme", ["Light","Dark"], index=0)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

# Language dict
T = {
    "English": {
        "dashboard": "📊 Dashboard",
        "features": "🔍 Feature Analysis",
        "prediction": "🔮 Prediction",
        "training": "⚡ Training",
        "rowscols": "Rows / Columns",
        "missing": "Missing Values",
        "summary": "Statistical Summary",
        "target": "Target Distribution",
        "top_corr": "Top Features Correlated with Target",
        "kpi": "Model KPIs",
        "cm": "Confusion Matrix",
        "roc": "ROC Curve",
        "pr": "Precision-Recall Curve",
        "fi": "Feature Importance",
        "pdp": "Partial Dependence Plot",
        "shap": "SHAP Force Plot",
        "upload": "Upload CSV/Excel for Prediction",
        "preview": "Preview of Uploaded Data",
        "results": "Prediction Results",
        "download": "Download Results",
        "train": "Retrain Models",
        "choose_models": "Select Models to Train",
    },
    "עברית": {
        "dashboard": "📊 לוח בקרה",
        "features": "🔍 ניתוח מאפיינים",
        "prediction": "🔮 חיזוי",
        "training": "⚡ אימון מודלים",
        "rowscols": "שורות / עמודות",
        "missing": "ערכים חסרים",
        "summary": "סטטיסטיקות",
        "target": "התפלגות יעד",
        "top_corr": "קורלציה מול היעד",
        "kpi": "מדדי המודל",
        "cm": "מטריצת בלבול",
        "roc": "עקומת ROC",
        "pr": "עקומת Precision-Recall",
        "fi": "חשיבות מאפיינים",
        "pdp": "Partial Dependence Plot",
        "shap": "SHAP Force Plot",
        "upload": "העלה קובץ CSV/Excel לחיזוי",
        "preview": "תצוגה מקדימה",
        "results": "תוצאות חיזוי",
        "download": "הורד תוצאות",
        "train": "אימון מודלים",
        "choose_models": "בחר מודלים לאימון",
    }
}[lang]

# ==============================
# Load Data & Model
# ==============================
DATA_PATH = "data/parkinsons.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop("status", axis=1)
y = df["status"]

best_model = joblib.load("models/best_model.joblib")
with open("assets/metrics.json","r") as f:
    metrics = json.load(f)

# User can pick which model to display
available_models = list(metrics.keys())
selected_model = st.sidebar.selectbox("Select Model to Display", available_models)
model_metrics = metrics[selected_model]
model = best_model  # לצורך פשטות נטען רק את הטוב ביותר ששמור כרגע

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    T["dashboard"], 
    T["features"], 
    T["prediction"], 
    T["training"]
])

# --- Tab 1: Dashboard
with tab1:
    st.header(T["dashboard"])

    # KPIs
    kpi_cols = st.columns(5)
    for i, k in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
        if k in model_metrics:
            kpi_cols[i].metric(k.capitalize(), f"{model_metrics[k]:.3f}")

    # Confusion Matrix (Plotly)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Blues")
    st.subheader(T["cm"])
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    y_pred_prob = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={model_metrics['roc_auc']:.2f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
    st.subheader(T["roc"])
    st.plotly_chart(fig, use_container_width=True)

    # PR Curve
    prec, rec, _ = precision_recall_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve"))
    st.subheader(T["pr"])
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Feature Analysis
with tab2:
    st.header(T["features"])

    # Feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader(T["fi"])
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig = px.bar(importances, orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    # PDP
    try:
        st.subheader(T["pdp"])
        fig, ax = plt.subplots()
        PartialDependenceDisplay.from_estimator(model, X, [0], ax=ax)
        st.pyplot(fig)
    except Exception:
        st.info("PDP not available")

    # SHAP
    try:
        st.subheader(T["shap"])
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X.iloc[:50])
        st.pyplot(shap.plots.force(shap_values[0], matplotlib=True, show=False))
    except Exception:
        st.warning("SHAP not available")

# --- Tab 3: Prediction
with tab3:
    st.header(T["prediction"])
    file = st.file_uploader(T["upload"], type=["csv","xlsx"])
    if file:
        if file.name.endswith(".csv"):
            new_df = pd.read_csv(file)
        else:
            new_df = pd.read_excel(file)

        st.write(T["preview"])
        st.dataframe(new_df.head())

        probs = model.predict_proba(new_df)[:,1]
        preds = (probs >= threshold).astype(int)
        new_df["Probability"] = probs
        new_df["Prediction"] = preds

        st.write(T["results"])
        st.table(new_df["Prediction"].value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

        cm = confusion_matrix(y[:len(preds)], preds)
        fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Oranges")
        st.plotly_chart(fig)

        st.download_button("📥 " + T["download"] + " (CSV)", new_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- Tab 4: Training
with tab4:
    st.header(T["training"])
    model_choices = st.multiselect(T["choose_models"], 
        ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"],
        default=["RandomForest","XGBoost"])
    rf_trees = st.slider("RandomForest Trees", 50, 500, 200, 50)
    xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.5, 0.1, 0.01)

    if st.button(T["train"]):
        config = {"models": model_choices, "params": {"rf_trees": rf_trees, "xgb_lr": xgb_lr}}
        os.makedirs("assets", exist_ok=True)
        with open("assets/config.json","w") as f:
            json.dump(config, f, indent=4)
        runpy.run_path("app/model_pipeline.py")
        st.success("✅ Training complete! Reload dashboard for updates.")
