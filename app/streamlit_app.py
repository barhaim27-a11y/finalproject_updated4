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

# --- Tab 2: Dashboard
with tab_dash:
    st.header("📈 Interactive Dashboard – Compare Models")

    model_options = ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"]
    chosen_models = st.multiselect("בחר מודלים להשוואה", model_options, default=["RandomForest","XGBoost"])

    st.subheader("⚙️ Hyperparameters")
    params = {}
    if "RandomForest" in chosen_models:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5)
        }
    if "XGBoost" in chosen_models:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50)
        }
    if "SVM" in chosen_models:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1)
        }

    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        trained_models = {}
        metrics_comp = {}

        for m in chosen_models:
            if m == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=params[m]["n_estimators"], 
                    max_depth=params[m]["max_depth"], 
                    random_state=42
                )
            elif m == "XGBoost":
                model = xgb.XGBClassifier(
                    eval_metric="logloss", 
                    n_estimators=params[m]["n_estimators"], 
                    learning_rate=params[m]["learning_rate"], 
                    random_state=42
                )
            elif m == "SVM":
                model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=params[m]["C"], probability=True, kernel="rbf"))])
            elif m == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
            elif m == "KNN":
                model = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(random_state=42)
            elif m == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            elif m == "NeuralNet":
                model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))])
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

        st.subheader("📊 Metrics Comparison")
        df_comp = pd.DataFrame(metrics_comp).T.sort_values("roc_auc", ascending=False)
        df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
        df_comp_display = df_comp.copy()
        df_comp_display.iloc[0, df_comp_display.columns.get_loc("Rank")] = "🏆 1"
        st.dataframe(df_comp_display)

        st.subheader("ROC Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Precision-Recall Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=m))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Models
with tab2:
    st.header("🤖 Model Training & Comparison")
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    df_metrics = df_metrics.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df_metrics.insert(0, "Rank", df_metrics.index + 1)
    best_name = df_metrics.iloc[0]["Model"]
    df_metrics.loc[0, "Model"] = f"🏆 {best_name}"

    st.subheader("📊 Model Ranking")
    st.dataframe(df_metrics)

    st.subheader("🏆 Best Model Results")
    best_row = df_metrics.iloc[0]
    st.table(pd.DataFrame(best_row).T)

    # Confusion Matrix (Plotly Heatmap)
    y_pred = safe_predict(best_model, X)
    y_pred_prob = safe_predict_proba(best_model, X)[:, 1]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    
        fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Healthy", "Parkinson’s"],
        y=["Healthy", "Parkinson’s"],
        colorscale="Oranges",
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC={roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curve")
    st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR Curve"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

    # Learning Curve
    st.subheader("Learning Curve – Best Model")
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1),
                             mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(test_scores, axis=1),
                             mode="lines+markers", name="Validation"))
    fig.update_layout(title="Learning Curve", xaxis_title="Training Samples", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative Gain Curve
    st.subheader("Cumulative Gain Curve")
    gains = np.cumsum(np.sort(y_pred_prob)[::-1]) / sum(y)
    percents = np.linspace(0, 1, len(gains))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents, y=gains, mode="lines", name="Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig.update_layout(title="Cumulative Gain Curve", xaxis_title="Proportion of Sample", yaxis_title="Proportion of Positives")
    st.plotly_chart(fig, use_container_width=True)

    # Lift Curve
    st.subheader("Lift Curve")
    lift = gains / percents
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents[1:], y=lift[1:], mode="lines", name="Lift"))
    fig.update_layout(title="Lift Curve", xaxis_title="Proportion of Sample", yaxis_title="Lift")
    st.plotly_chart(fig, use_container_width=True)

    # KS Curve
    st.subheader("KS Curve")
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    ks_stat = max(tpr - fpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=tpr, mode="lines", name="TPR"))
    fig.add_trace(go.Scatter(x=thresholds, y=fpr, mode="lines", name="FPR"))
    fig.update_layout(title=f"KS Curve (KS={ks_stat:.2f})", xaxis_title="Threshold", yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)



# --- Tab 4: Prediction
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

            cm = confusion_matrix(preds, new_df["Prediction"])
            fig = ff.create_annotated_heatmap(cm, x=["Healthy","Parkinson’s"], y=["Healthy","Parkinson’s"], colorscale="Oranges")
            st.plotly_chart(fig)

            st.download_button("📥 Download Predictions (CSV)", new_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- Tab 5: Train New Model
with tab4:
    st.header("⚡ Train New Model")
    model_choices = st.multiselect("Select Models", ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"], default=["RandomForest","XGBoost"])
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
