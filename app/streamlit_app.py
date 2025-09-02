import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json, io, shutil

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)

st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

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

def risk_label(prob, threshold=0.5):
    if prob < 0.3:
        return "🟢 Low"
    elif prob < 0.7:
        return "🟡 Medium"
    else:
        return "🔴 High"

def decision_text(prob, threshold=0.5):
    decision = "Positive (Parkinson’s)" if prob >= threshold else "Negative (Healthy)"
    return f"ההסתברות היא {prob*100:.1f}%, הסיווג עם הסף {threshold:.2f} הוא {decision}"

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

    # --- סטטיסטיקות ---
    st.subheader("Dataset Info & Statistics")
    st.write(f"🔹 Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning("Missing Values detected:")
        st.dataframe(missing[missing > 0])
    else:
        st.success("No missing values ✅")

    st.write("🔹 Statistical Summary")
    st.dataframe(df.describe().T)

    st.write("🔹 Target Distribution (Counts)")
    st.table(y.value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

    st.write("🔹 Top Features Correlated with Target")
    corr_target = df.corr()["status"].abs().sort_values(ascending=False)[1:6]
    st.table(corr_target)

    # --- גרפים מוכנים מתיקיית eda ---
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
            with st.expander(title, expanded=True):
                st.image(path, use_column_width=True)

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")

    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(df_metrics)

    # === KPIs Dashboard ===
    kpi_cols = st.columns(5)
    best_row = df_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    for i, k in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
        if k in best_row:
            kpi_cols[i].metric(k.capitalize(), f"{best_row[k]:.3f}")

    # === Bar chart ROC-AUC ===
    if "roc_auc" in df_metrics.columns:
        st.bar_chart(df_metrics.set_index("Model")["roc_auc"])

    # Best Model
    best_name = best_row["Model"]
    st.success(f"🏆 Best Model: {best_name}")

    # === ROC & PR Comparison for all models ===
    st.subheader("ROC & PR Curves (All Models)")
    fig, ax = plt.subplots()
    for _, row in df_metrics.iterrows():
        model_path = "models/best_model.joblib" if row["Model"] == best_name else None
        if model_path and os.path.exists(model_path):
            model = best_model
            y_pred_prob = safe_predict_proba(model, X)[:,1]
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            ax.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['roc_auc']:.2f})")
    ax.plot([0,1],[0,1],'k--')
    ax.legend()
    st.pyplot(fig)

    # Confusion Matrix (Best Model)
    st.subheader("Confusion Matrix (Best Model)")
    y_pred = safe_predict(best_model, X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Parkinson’s"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    st.pyplot(fig)

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        st.subheader("Feature Importance")
        importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=importances, y=importances.index, ax=ax)
        st.pyplot(fig)

    # SHAP Summary Plot
    shap_path = os.path.join("assets","shap_summary.png")
    if os.path.exists(shap_path):
        st.subheader("Explainability (SHAP)")
        st.image(shap_path, caption="SHAP Feature Importance")

# --- Tab 3: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV"])
    
    if option=="Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
            st.progress(prob)
            st.write(risk_label(prob, threshold))
            st.info(decision_text(prob, threshold))
    
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            probs = safe_predict_proba(best_model, new_df)[:,1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = probs
            new_df["Prediction"] = preds
            new_df["risk_label"] = [risk_label(p, threshold) for p in probs]
            new_df["decision_text"] = [decision_text(p, threshold) for p in probs]
            st.dataframe(new_df.head())

            # Summary counts
            st.write("🔹 Prediction Summary")
            st.table(new_df["Prediction"].value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    st.write("בחר אילו מודלים לאמן והגדר היפר-פרמטרים בסיסיים:")

    model_choices = st.multiselect("בחר מודלים", ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"], default=["RandomForest","XGBoost"])
    rf_trees = st.slider("RandomForest Trees", 50, 500, 200, 50)
    xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.5, 0.1, 0.01)

    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())

        if st.button("Retrain Models"):
            new_path = "data/new_train.csv"
            new_df.to_csv(new_path, index=False)
            # ⚠️ כאן ניתן לשלב קריאה ל-pipeline מעודכן שיקח את הפרמטרים
            runpy.run_path("app/model_pipeline.py")

            st.success("✅ Models retrained! Reload the app to see updates.")
