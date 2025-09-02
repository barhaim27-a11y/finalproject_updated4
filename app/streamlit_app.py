# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json, io, shutil

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

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

def export_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf

# ==============================
# Load dataset
# ==============================
DATA_PATH = "data/parkinsons.csv"
df = pd.read_csv(DATA_PATH)
if "name" in df.columns:
    df = df.drop(columns=["name"])
X = df.drop("status", axis=1)
y = df["status"]

# ==============================
# Load model + metrics once
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
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.download_button("📥 Download Dataset (CSV)", df.to_csv(index=False).encode("utf-8"), "dataset.csv", "text/csv")

    st.write("### Statistical Summary")
    st.dataframe(df.describe().T)

    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="status", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.write("### PCA Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_title("PCA Projection")
    st.pyplot(fig)

    st.write("### t-SNE Visualization (sampled)")
    sample_size = min(300, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X_sample)
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample, cmap="coolwarm", alpha=0.7)
    ax.set_title("t-SNE Projection (sample of 300)")
    st.pyplot(fig)

# --- Tab 2: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    st.dataframe(df_metrics)

    # KPIs
    cols = st.columns(5)
    for i, k in enumerate(["accuracy","precision","recall","f1","roc_auc"]):
        cols[i].metric(k.capitalize(), f"{df_metrics.iloc[0][k]:.3f}")

    # Bar chart
    st.bar_chart(df_metrics.set_index("Model")["roc_auc"])

    # Best Model
    best_name = df_metrics.sort_values("roc_auc", ascending=False).iloc[0]["Model"]
    st.success(f"🏆 Best Model: {best_name}")

    # Confusion Matrix
    y_pred = safe_predict(best_model, X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Parkinson’s"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    y_pred_prob = safe_predict_proba(best_model, X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
    ax.plot([0,1],[0,1],'k--')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y, y_pred_prob)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="Precision-Recall Curve")
    ax.legend()
    st.pyplot(fig)

    # SHAP Summary Plot
    shap_path = os.path.join("assets","shap_summary.png")
    if os.path.exists(shap_path):
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

# --- Tab 4: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())

        if st.button("Retrain Models"):
            new_path = "data/new_train.csv"
            new_df.to_csv(new_path, index=False)
            runpy.run_path("app/model_pipeline.py")

            st.success("✅ Models retrained! Reload the app to see updates.")
