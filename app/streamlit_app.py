import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json, runpy, shap, shutil
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.figure_factory as ff
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

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Parkinson’s ML App", page_icon="🧠", layout="wide")

st.sidebar.title("⚙️ Settings")
threshold_global = st.sidebar.slider("Decision Threshold (Global)", 0.0, 1.0, 0.5, 0.01)

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
tab1, tab_dash, tab2, tab3, tab5, tab4 = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model"
])

# --- Tab 1: Data & EDA
with tab1:
    st.header("📊 Data & Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info & Statistics")
    st.write(f"🔹 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.describe().T)
    st.table(y.value_counts().rename({0:"Healthy",1:"Parkinson’s"}))

# --- Tab 2: Dashboard
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")
    st.info("בחר מודלים להשוואה ואמן על דאטה קיים")

    model_options = ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"]
    chosen_models = st.multiselect("בחר מודלים להשוואה", model_options, default=["RandomForest","XGBoost"])

    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        trained_models, metrics_comp = {}, {}
        for m in chosen_models:
            if m == "RandomForest":
                model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
            elif m == "XGBoost":
                model = xgb.XGBClassifier(eval_metric="logloss", n_estimators=200, learning_rate=0.1, random_state=42)
            elif m == "SVM":
                model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=1.0, probability=True))])
            elif m == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
            elif m == "KNN":
                model = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(random_state=42)
            elif m == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            elif m == "NeuralNet":
                model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500))])
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_val = roc_auc_score(y_test, y_proba)

            trained_models[m] = model
            metrics_comp[m] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc_val}

        st.session_state.trained_models = trained_models
        st.subheader("📊 Metrics Comparison")
        st.dataframe(pd.DataFrame(metrics_comp).T)

# --- Tab 3: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
    df_metrics = df_metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    st.dataframe(df_metrics)

# --- Tab 4: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)

    # ✅ בחירת מודל לניבוי
    available_models = {"Best Model": best_model}
    if "trained_models" in st.session_state:
        available_models.update(st.session_state.trained_models)

    model_choice = st.selectbox("בחר מודל לניבוי", list(available_models.keys()))
    model = available_models[model_choice]

    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])
    if option == "Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])
        if st.button("Predict Sample"):
            prob = safe_predict_proba(model, sample)[0,1]
            st.write(f"Probability: {prob*100:.1f}%")
    else:
        file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
        if file:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)
            probs = safe_predict_proba(model, new_df)[:,1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = (probs*100).round(1)
            new_df["Prediction"] = preds
            st.dataframe(new_df.head())

# --- Tab 5: Test Evaluation
with tab5:
    st.header("🧪 Model Evaluation on External Test Set")
    file = st.file_uploader("Upload Test Set (CSV with 'status' column)", type=["csv"], key="testset")
    if file:
        test_df = pd.read_csv(file)
        if "status" in test_df.columns:
            X_test = test_df.drop("status", axis=1)
            y_true = test_df["status"]
            y_pred = safe_predict(best_model, X_test)
            y_prob = safe_predict_proba(best_model, X_test)[:, 1]
            st.write("Accuracy:", accuracy_score(y_true, y_pred))
        else:
            st.error("❌ Must include 'status' column")

# --- Tab 6: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    model_choices = st.multiselect("Select Models", ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"], default=["RandomForest","XGBoost"])
    rf_trees = st.slider("RandomForest Trees", 50, 500, 200, 50)
    xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.5, 0.1, 0.01)

    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("📂 New Data Preview:", new_df.head())

        if st.button("Retrain Models"):
            # ✅ שילוב הדאטה עם המקורי
            combined_df = pd.concat([df, new_df], ignore_index=True)
            combined_df.to_csv("data/combined_train.csv", index=False)

            config = {"models": model_choices, "params": {"rf_trees": rf_trees, "xgb_lr": xgb_lr}}
            os.makedirs("assets", exist_ok=True)
            with open("assets/config.json","w") as f:
                json.dump(config, f, indent=4)
            runpy.run_path("app/model_pipeline.py")

            with open("assets/metrics.json","r") as f:
                new_metrics = json.load(f)

            st.session_state.new_metrics = new_metrics
            st.session_state.new_best_model = joblib.load("models/best_model.joblib")

            st.subheader("📊 New Training Results")
            comp_df = pd.DataFrame(new_metrics).T.reset_index().rename(columns={"index":"Model"})
            comp_df = comp_df.sort_values("roc_auc", ascending=False)
            st.dataframe(comp_df.style.highlight_max(axis=0, color="lightgreen"))

            # ✅ השוואה מול המודל הישן
            old_auc = max(metrics.values(), key=lambda m: m["roc_auc"])
            new_auc = max(new_metrics.values(), key=lambda m: m["roc_auc"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Old Best ROC-AUC", f"{old_auc:.3f}")
            with col2:
                st.metric("New Best ROC-AUC", f"{new_auc:.3f}")

            if new_auc > old_auc:
                st.success("🎉 המודל החדש עדיף על המודל הישן!")
                if st.button("🚀 Promote New Model"):
                    shutil.copy("models/best_model.joblib", "models/best_model_promoted.joblib")
                    with open("assets/metrics.json","w") as f:
                        json.dump(new_metrics, f)
                    st.success("✅ New model promoted as best model!")

            # ✅ ROC Curve comparison
            st.subheader("ROC Curve – Old vs New Best Model")
            y_pred_prob_old = safe_predict_proba(best_model, X)[:,1]
            y_pred_prob_new = safe_predict_proba(st.session_state.new_best_model, X)[:,1]
            fpr_old, tpr_old, _ = roc_curve(y, y_pred_prob_old)
            fpr_new, tpr_new, _ = roc_curve(y, y_pred_prob_new)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_old, y=tpr_old, mode="lines", name="Old Best"))
            fig.add_trace(go.Scatter(x=fpr_new, y=tpr_new, mode="lines", name="New Best"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
            st.plotly_chart(fig, use_container_width=True)
