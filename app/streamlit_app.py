import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, runpy, json, io, shutil
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
)
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

st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

# ==============================
# Helpers (עודכן ל-KeyError)
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
        runpy.run_path("model_pipeline.py")
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
    st.write("### Dataset Preview")
    st.dataframe(df.head())

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
    st.pyplot(fig)

    st.write("### t-SNE Visualization (sample of 300)")
    sample_size = min(300, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X_sample)
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample, cmap="coolwarm", alpha=0.7)
    st.pyplot(fig)

# --- Tab 2: Dashboard
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")

    model_options = ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"]
    chosen_models = st.multiselect("בחר מודלים להשוואה", model_options, default=["RandomForest","XGBoost"])

    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        metrics_comp = {}
        for m in chosen_models:
            if m == "RandomForest":
                model = RandomForestClassifier(n_estimators=200, random_state=42)
            elif m == "XGBoost":
                model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
            elif m == "SVM":
                model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))])
            elif m == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
            elif m == "KNN":
                model = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(random_state=42)
            elif m == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            elif m == "NeuralNet":
                model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=500, random_state=42))])
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            metrics_comp[m] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba)
            }

        st.subheader("📊 Metrics Comparison")
        st.dataframe(pd.DataFrame(metrics_comp).T)

        st.subheader("ROC Curves")
        fig, ax = plt.subplots()
        for m in chosen_models:
            model = None
            if m == "RandomForest":
                model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
            elif m == "XGBoost":
                model = xgb.XGBClassifier(eval_metric="logloss", random_state=42).fit(X_train, y_train)
            if model:
                y_proba = model.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                ax.plot(fpr, tpr, label=f"{m}")
        ax.plot([0,1],[0,1],'k--')
        ax.legend()
        st.pyplot(fig)

# --- Tab 3: Models
with tab2:
    st.header("🤖 Models")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Model","ROC-AUC"]).sort_values(by="ROC-AUC", ascending=False)
    st.dataframe(df_metrics)

    y_pred = safe_predict(best_model, X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy","Parkinson’s"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    st.pyplot(fig)

    y_pred_prob = safe_predict_proba(best_model, X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0,1],[0,1],'k--')
    ax.legend()
    st.pyplot(fig)

# --- Tab 4: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV"])

    if option=="Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0,1]
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
            st.dataframe(new_df.head())
            st.download_button("📥 Download Predictions (CSV)", new_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- Tab 5: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("New Data Preview:", new_df.head())
        if st.button("Retrain Models"):
            new_path = "data/new_train.csv"
            new_df.to_csv(new_path, index=False)
            runpy.run_path("model_pipeline.py")
            shutil.copy("models/best_model.joblib", "models/best_model_new.joblib")
            shutil.copy("assets/metrics.json", "assets/metrics_new.json")
            with open("assets/metrics_new.json","r") as f:
                new_metrics = json.load(f)
            st.dataframe(pd.DataFrame(new_metrics.items(), columns=["Model","ROC-AUC"]))
