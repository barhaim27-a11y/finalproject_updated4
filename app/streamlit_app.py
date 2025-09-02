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
# Demo Mode
demo_mode = st.sidebar.checkbox("Enable Demo Mode (load sample data)", value=False, key="demo_mode")

# Dataset Handling
dataset_mode = st.sidebar.radio(
    "Dataset Mode",
    ["Use Default Dataset", "Replace with Uploaded", "Merge with Uploaded"],
    index=0,
    key="dataset_mode"
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
if demo_mode:
    st.info("Demo Mode Enabled – using sample dataset ✅")
    df = pd.read_csv(DATA_PATH).sample(50, random_state=42)  # תת-סט קטן להדגמה
else:
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

# Model Last Updated
last_updated_path = "assets/last_updated.txt"
if os.path.exists(last_updated_path):
    with open(last_updated_path, "r") as f:
        last_updated = f.read().strip()
else:
    last_updated = "Unknown"

st.sidebar.markdown(f"🕒 **Model Last Updated:** {last_updated}")

# ==============================
# Tabs
# ==============================
tab1, tab_dash, tab2, tab3, tab5, tab4, tab_hist, tab_explain, tab_about = st.tabs([
    "📊 Data & EDA", 
    "📈 Dashboard",
    "🤖 Models", 
    "🔮 Prediction", 
    "🧪 Test Evaluation",
    "⚡ Train New Model",
    "🕑 Model History",
    "🧠 Explainability",
    "ℹ️ About"
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

    # --- בחירת מודלים ---
    model_options = [
        "LogisticRegression","RandomForest","SVM",
        "KNN","XGBoost","LightGBM","CatBoost","NeuralNet"
    ]
    chosen_models = st.multiselect(
        "בחר מודלים להשוואה", 
        model_options, 
        default=["RandomForest","XGBoost"]
    )
    
    # --- היפר-פרמטרים ---
    st.subheader("⚙️ Hyperparameters")
    params = {}

    if "LogisticRegression" in chosen_models:
        params["LogisticRegression"] = {
            "C": st.slider("LogReg: Regularization C", 0.01, 10.0, 1.0, 0.1, key="logreg_c_dash"),
            "max_iter": st.slider("LogReg: Max Iterations", 100, 2000, 500, 100, key="logreg_iter_dash")
        }

    if "RandomForest" in chosen_models:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50, key="rf_trees_dash"),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5, key="rf_depth_dash"),
            "min_samples_split": st.slider("RF: Min Samples Split", 2, 20, 2, key="rf_split_dash"),
            "min_samples_leaf": st.slider("RF: Min Samples Leaf", 1, 20, 1, key="rf_leaf_dash")
        }

    if "XGBoost" in chosen_models:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr_dash"),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50, key="xgb_estimators_dash"),
            "max_depth": st.slider("XGB: Max Depth", 2, 20, 6, key="xgb_depth_dash"),
            "subsample": st.slider("XGB: Subsample", 0.5, 1.0, 1.0, 0.05, key="xgb_subsample_dash"),
            "colsample_bytree": st.slider("XGB: Colsample by Tree", 0.5, 1.0, 1.0, 0.05, key="xgb_colsample_dash")
        }

    if "LightGBM" in chosen_models:
        params["LightGBM"] = {
            "n_estimators": st.slider("LGBM: Estimators", 50, 500, 200, 50, key="lgb_estimators_dash"),
            "learning_rate": st.slider("LGBM: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lgb_lr_dash"),
            "num_leaves": st.slider("LGBM: Num Leaves", 10, 200, 31, key="lgb_leaves_dash"),
            "max_depth": st.slider("LGBM: Max Depth", -1, 20, -1, key="lgb_depth_dash")
        }

    if "CatBoost" in chosen_models:
        params["CatBoost"] = {
            "iterations": st.slider("CatBoost: Iterations", 100, 2000, 500, 100, key="cat_iters_dash"),
            "depth": st.slider("CatBoost: Depth", 2, 12, 6, key="cat_depth_dash"),
            "learning_rate": st.slider("CatBoost: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="cat_lr_dash")
        }

    if "SVM" in chosen_models:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1, key="svm_c_dash"),
            "kernel": st.selectbox("SVM: Kernel", ["rbf", "linear", "poly"], index=0, key="svm_kernel_dash"),
            "gamma": st.selectbox("SVM: Gamma", ["scale", "auto"], index=0, key="svm_gamma_dash")
        }

    if "KNN" in chosen_models:
        params["KNN"] = {
            "n_neighbors": st.slider("KNN: Neighbors", 1, 20, 5, key="knn_neighbors_dash"),
            "weights": st.selectbox("KNN: Weights", ["uniform", "distance"], index=0, key="knn_weights_dash")
        }

    if "NeuralNet" in chosen_models:
        params["NeuralNet"] = {
            "hidden_layer_sizes": st.text_input("NN: Hidden Layers (comma-separated)", "64,32", key="nn_layers_dash"),
            "activation": st.selectbox("NN: Activation", ["relu", "tanh", "logistic"], index=0, key="nn_activation_dash"),
            "max_iter": st.slider("NN: Max Iterations", 100, 2000, 500, 100, key="nn_iter_dash")
        }

    # --- הפעלה ---
    if st.button("🚀 Run Comparison"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        trained_models = {}
        metrics_comp = {}

        for m in chosen_models:
            model = None

            if m == "LogisticRegression":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=params[m]["C"],
                        max_iter=params[m]["max_iter"]
                    ))
                ])
            elif m == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=params[m]["n_estimators"],
                    max_depth=params[m]["max_depth"],
                    min_samples_split=params[m]["min_samples_split"],
                    min_samples_leaf=params[m]["min_samples_leaf"],
                    random_state=42
                )
            elif m == "XGBoost":
                model = xgb.XGBClassifier(
                    eval_metric="logloss",
                    n_estimators=params[m]["n_estimators"],
                    learning_rate=params[m]["learning_rate"],
                    max_depth=params[m]["max_depth"],
                    subsample=params[m]["subsample"],
                    colsample_bytree=params[m]["colsample_bytree"],
                    random_state=42
                )
            elif m == "LightGBM":
                model = lgb.LGBMClassifier(
                    n_estimators=params[m]["n_estimators"],
                    learning_rate=params[m]["learning_rate"],
                    num_leaves=params[m]["num_leaves"],
                    max_depth=params[m]["max_depth"],
                    random_state=42
                )
            elif m == "CatBoost":
                model = CatBoostClassifier(
                    iterations=params[m]["iterations"],
                    depth=params[m]["depth"],
                    learning_rate=params[m]["learning_rate"],
                    verbose=0,
                    random_state=42
                )
            elif m == "SVM":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(
                        C=params[m]["C"],
                        kernel=params[m]["kernel"],
                        gamma=params[m]["gamma"],
                        probability=True
                    ))
                ])
            elif m == "KNN":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(
                        n_neighbors=params[m]["n_neighbors"],
                        weights=params[m]["weights"]
                    ))
                ])
            elif m == "NeuralNet":
                hidden_layers = tuple(map(int, params[m]["hidden_layer_sizes"].split(",")))
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=params[m]["activation"],
                        max_iter=params[m]["max_iter"],
                        random_state=42
                    ))
                ])

            # --- אימון והערכת ביצועים ---
            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc_val = roc_auc_score(y_test, y_proba)

                trained_models[m] = model
                metrics_comp[m] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": auc_val
                }

        # --- טבלת השוואה ---
        st.subheader("📊 Metrics Comparison")
        df_comp = pd.DataFrame(metrics_comp).T.sort_values("roc_auc", ascending=False)
        df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
        df_comp_display = df_comp.copy()
        df_comp_display.iloc[0, df_comp_display.columns.get_loc("Rank")] = "🏆 1"
        st.dataframe(df_comp_display)

        # --- ROC Curves ---
        st.subheader("ROC Curves")
        fig = go.Figure()
        for m, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
        st.plotly_chart(fig, use_container_width=True)

        # --- Precision-Recall Curves ---
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
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

if st.button("📄 Download Full Report (PDF)"):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 14)
    c.drawString(50, 750, "Parkinson’s ML Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, 720, f"Last Updated: {last_updated}")
    c.drawString(50, 700, f"Best Model: {best_name}")
    c.drawString(50, 680, f"Metrics: {json.dumps(metrics[best_name], indent=2)}")

    c.save()
    st.download_button("📥 Download Report PDF", pdf_buffer.getvalue(),
                       file_name="report.pdf", mime="application/pdf")

# --- Tab 4: Prediction
with tab3:
    st.header("🔮 Prediction")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, threshold_global, 0.01)
    
    option = st.radio("Choose input type:", ["Manual Input","Upload CSV/Excel"])

    # --- Manual Input
    if option == "Manual Input":
        inputs = {col: st.number_input(col, float(X[col].mean())) for col in X.columns}
        sample = pd.DataFrame([inputs])

        if st.button("Predict Sample"):
            prob = safe_predict_proba(best_model, sample)[0, 1]
            pred = int(prob >= threshold)

            st.subheader("🧾 Prediction Result")
            if pred == 1:
                st.markdown(f"""
                <div style="padding:15px; background-color:#ffe6e6; border-radius:10px; text-align:center">
                    <h2 style="color:#cc0000">🔴 Parkinson’s Detected</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="padding:15px; background-color:#e6ffe6; border-radius:10px; text-align:center">
                    <h2 style="color:#009900">🟢 Healthy</h2>
                    <p>Probability: <b>{prob*100:.1f}%</b> (Threshold={threshold:.2f})</p>
                </div>
                """, unsafe_allow_html=True)

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text": "Probability of Parkinson’s"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if pred == 1 else "green"},
                    "steps": [
                        {"range": [0, 30], "color": "#e6ffe6"},
                        {"range": [30, 70], "color": "#fff5e6"},
                        {"range": [70, 100], "color": "#ffe6e6"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    # --- File Upload
    elif option == "Upload CSV/Excel":
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if file:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            else:
                new_df = pd.read_excel(file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(new_df.head())

            probs = safe_predict_proba(best_model, new_df)[:, 1]
            preds = (probs >= threshold).astype(int)
            new_df["Probability"] = (probs*100).round(1)
            new_df["Prediction"] = preds

            # עמודת תוצאה ידידותית
            new_df["Result"] = [
                f"🟢 Healthy ({p:.1f}%)" if pred == 0 else f"🔴 Parkinson’s ({p:.1f}%)"
                for pred, p in zip(preds, new_df["Probability"])
            ]

            st.subheader("📊 Prediction Summary")
            summary = pd.Series(preds).value_counts().rename({0: "Healthy 🟢", 1: "Parkinson’s 🔴"})
            st.table(summary)

            st.subheader("Detailed Results")
            st.dataframe(new_df[["Result", "Probability", "Prediction"]].head(20))

            # הורדות
            st.download_button(
                "📥 Download Predictions (CSV)",
                new_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )
            new_df.to_excel("predictions.xlsx", index=False)
            with open("predictions.xlsx", "rb") as f:
                st.download_button(
                    "📥 Download Predictions (Excel)",
                    f,
                    "predictions.xlsx",
                    "application/vnd.ms-excel"
                )

                
# --- Tab 5: Test Evaluation
with tab5:
    st.header("🧪 Model Evaluation on External Test Set")

    file = st.file_uploader("Upload Test Set (CSV with 'status' column)", type=["csv"], key="testset")
    if file:
        test_df = pd.read_csv(file)
        if "status" not in test_df.columns:
            st.error("❌ The test set must include a 'status' column with true labels (0=Healthy, 1=Parkinson’s).")
        else:
            X_test = test_df.drop("status", axis=1)
            y_true = test_df["status"]

            y_pred = safe_predict(best_model, X_test)
            y_prob = safe_predict_proba(best_model, X_test)[:, 1]

            # 📊 Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc_val = roc_auc_score(y_true, y_prob)

            st.subheader("📊 Metrics")
            metrics_df = pd.DataFrame([{
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc_val
            }])
            st.dataframe(metrics_df.style.format("{:.3f}"))

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted Healthy", "Predicted Parkinson’s"],
                y=["True Healthy", "True Parkinson’s"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}"
            ))
            fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC={auc_val:.2f})"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            test_df["Predicted"] = y_pred
            test_df["Probability"] = y_prob
            st.download_button("📥 Download Predictions (CSV)", test_df.to_csv(index=False).encode("utf-8"), "test_results.csv", "text/csv")

            test_df.to_excel("test_results.xlsx", index=False)
            with open("test_results.xlsx","rb") as f:
                st.download_button("📥 Download Predictions (Excel)", f, "test_results.xlsx", "application/vnd.ms-excel")



# --- Tab 6: Train New Model
with tab4:
    st.header("⚡ Train New Model")

    st.markdown("העלה דאטה חדש לאימון, בחר מודלים והגדר פרמטרים – נבצע השוואה מול המודל הנוכחי הטוב ביותר.")

    # ✅ בחירת מודלים
    model_choices = st.multiselect(
        "Select Models",
        ["LogisticRegression","RandomForest","SVM","KNN","XGBoost","LightGBM","CatBoost","NeuralNet"],
        default=["RandomForest","XGBoost"]
    )

    # ✅ פרמטרים לכל מודל (keys ייחודיים ללשונית הזאת)
    st.subheader("⚙️ Hyperparameters")
    params = {}

    if "LogisticRegression" in model_choices:
        params["LogisticRegression"] = {
            "C": st.slider("LogReg: Regularization C", 0.01, 10.0, 1.0, 0.1, key="logreg_c_train"),
            "max_iter": st.slider("LogReg: Max Iterations", 100, 2000, 500, 100, key="logreg_iter_train")
        }

    if "RandomForest" in model_choices:
        params["RandomForest"] = {
            "n_estimators": st.slider("RF: Number of Trees", 50, 500, 200, 50, key="rf_trees_train"),
            "max_depth": st.slider("RF: Max Depth", 2, 20, 5, key="rf_depth_train"),
            "min_samples_split": st.slider("RF: Min Samples Split", 2, 20, 2, key="rf_split_train"),
            "min_samples_leaf": st.slider("RF: Min Samples Leaf", 1, 20, 1, key="rf_leaf_train")
        }

    if "XGBoost" in model_choices:
        params["XGBoost"] = {
            "learning_rate": st.slider("XGB: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="xgb_lr_train"),
            "n_estimators": st.slider("XGB: Estimators", 50, 500, 200, 50, key="xgb_estimators_train"),
            "max_depth": st.slider("XGB: Max Depth", 2, 20, 6, key="xgb_depth_train"),
            "subsample": st.slider("XGB: Subsample", 0.5, 1.0, 1.0, 0.05, key="xgb_subsample_train"),
            "colsample_bytree": st.slider("XGB: Colsample by Tree", 0.5, 1.0, 1.0, 0.05, key="xgb_colsample_train")
        }

    if "LightGBM" in model_choices:
        params["LightGBM"] = {
            "n_estimators": st.slider("LGBM: Estimators", 50, 500, 200, 50, key="lgb_estimators_train"),
            "learning_rate": st.slider("LGBM: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lgb_lr_train"),
            "num_leaves": st.slider("LGBM: Num Leaves", 10, 200, 31, key="lgb_leaves_train"),
            "max_depth": st.slider("LGBM: Max Depth", -1, 20, -1, key="lgb_depth_train")
        }

    if "CatBoost" in model_choices:
        params["CatBoost"] = {
            "iterations": st.slider("CatBoost: Iterations", 100, 2000, 500, 100, key="cat_iters_train"),
            "depth": st.slider("CatBoost: Depth", 2, 12, 6, key="cat_depth_train"),
            "learning_rate": st.slider("CatBoost: Learning Rate", 0.01, 0.5, 0.1, 0.01, key="cat_lr_train")
        }

    if "SVM" in model_choices:
        params["SVM"] = {
            "C": st.slider("SVM: Regularization C", 0.01, 10.0, 1.0, 0.1, key="svm_c_train"),
            "kernel": st.selectbox("SVM: Kernel", ["rbf", "linear", "poly"], index=0, key="svm_kernel_train"),
            "gamma": st.selectbox("SVM: Gamma", ["scale", "auto"], index=0, key="svm_gamma_train")
        }

    if "KNN" in model_choices:
        params["KNN"] = {
            "n_neighbors": st.slider("KNN: Neighbors", 1, 20, 5, key="knn_neighbors_train"),
            "weights": st.selectbox("KNN: Weights", ["uniform", "distance"], index=0, key="knn_weights_train")
        }

    if "NeuralNet" in model_choices:
        params["NeuralNet"] = {
            "hidden_layer_sizes": st.text_input("NN: Hidden Layers (comma-separated)", "64,32", key="nn_layers_train"),
            "activation": st.selectbox("NN: Activation", ["relu", "tanh", "logistic"], index=0, key="nn_activation_train"),
            "max_iter": st.slider("NN: Max Iterations", 100, 2000, 500, 100, key="nn_iter_train")
        }

    # ✅ קובץ דאטה חדש
    file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="newtrain")
    if file:
        new_df = pd.read_csv(file)
        st.write("📂 New Data Preview:", new_df.head())

        if st.button("🚀 Retrain Models"):
            # 🟢 שילוב הדאטה החדש עם המקורי
            combined_df = pd.concat([df, new_df], ignore_index=True)
            X_combined = combined_df.drop("status", axis=1)
            y_combined = combined_df["status"]

            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
            )

            trained_models = {}
            metrics_comp = {}

            # 🟢 לולאת אימון
            for m in model_choices:
                model = None

                if m == "LogisticRegression":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(
                            C=params[m]["C"],
                            max_iter=params[m]["max_iter"]
                        ))
                    ])
                elif m == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=params[m]["n_estimators"],
                        max_depth=params[m]["max_depth"],
                        min_samples_split=params[m]["min_samples_split"],
                        min_samples_leaf=params[m]["min_samples_leaf"],
                        random_state=42
                    )
                elif m == "XGBoost":
                    model = xgb.XGBClassifier(
                        eval_metric="logloss",
                        n_estimators=params[m]["n_estimators"],
                        learning_rate=params[m]["learning_rate"],
                        max_depth=params[m]["max_depth"],
                        subsample=params[m]["subsample"],
                        colsample_bytree=params[m]["colsample_bytree"],
                        random_state=42
                    )
                elif m == "LightGBM":
                    model = lgb.LGBMClassifier(
                        n_estimators=params[m]["n_estimators"],
                        learning_rate=params[m]["learning_rate"],
                        num_leaves=params[m]["num_leaves"],
                        max_depth=params[m]["max_depth"],
                        random_state=42
                    )
                elif m == "CatBoost":
                    model = CatBoostClassifier(
                        iterations=params[m]["iterations"],
                        depth=params[m]["depth"],
                        learning_rate=params[m]["learning_rate"],
                        verbose=0,
                        random_state=42
                    )
                elif m == "SVM":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", SVC(
                            C=params[m]["C"],
                            kernel=params[m]["kernel"],
                            gamma=params[m]["gamma"],
                            probability=True
                        ))
                    ])
                elif m == "KNN":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", KNeighborsClassifier(
                            n_neighbors=params[m]["n_neighbors"],
                            weights=params[m]["weights"]
                        ))
                    ])
                elif m == "NeuralNet":
                    hidden_layers = tuple(map(int, params[m]["hidden_layer_sizes"].split(",")))
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", MLPClassifier(
                            hidden_layer_sizes=hidden_layers,
                            activation=params[m]["activation"],
                            max_iter=params[m]["max_iter"],
                            random_state=42
                        ))
                    ])

                if model is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc_val = roc_auc_score(y_test, y_proba)

                    trained_models[m] = model
                    metrics_comp[m] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": auc_val
                    }

            # 🟢 שמירה ב־session_state
            st.session_state.trained_models = trained_models

            # 🟢 תוצאות
            st.subheader("📊 New Training Results")
            df_comp = pd.DataFrame(metrics_comp).T.sort_values("roc_auc", ascending=False)
            df_comp.insert(0, "Rank", range(1, len(df_comp)+1))
            df_comp_display = df_comp.copy()
            df_comp_display.iloc[0, df_comp_display.columns.get_loc("Rank")] = "🏆 1"
            st.dataframe(df_comp_display)

            # 🟢 השוואה מול המודל הישן
            st.subheader("📈 Comparison with Old Best Model")
            y_pred_old = safe_predict(best_model, X_test)
            y_proba_old = safe_predict_proba(best_model, X_test)[:, 1]
            old_auc = roc_auc_score(y_test, y_proba_old)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Old Best ROC-AUC", f"{old_auc:.3f}")
            with col2:
                st.metric("New Best ROC-AUC", f"{df_comp['roc_auc'].iloc[0]:.3f}")

            # 🟢 ROC Curve comparison
            fig = go.Figure()
            fpr_old, tpr_old, _ = roc_curve(y_test, y_proba_old)
            fig.add_trace(go.Scatter(x=fpr_old, y=tpr_old, mode="lines", name="Old Best"))
            for m in df_comp.index:
                y_proba_new = trained_models[m].predict_proba(X_test)[:, 1]
                fpr_new, tpr_new, _ = roc_curve(y_test, y_proba_new)
                fig.add_trace(go.Scatter(
                    x=fpr_new, y=tpr_new, mode="lines",
                    name=f"{m} (AUC={metrics_comp[m]['roc_auc']:.2f})"
                ))
            fig.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(dash="dash"), name="Random"
            ))
            st.plotly_chart(fig, use_container_width=True)

            # 🟢 Promote option
            from datetime import datetime
            with open("assets/last_updated.txt", "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            best_new_model = df_comp.index[0]
            if df_comp["roc_auc"].iloc[0] > old_auc:
                st.success(f"🎉 המודל החדש {best_new_model} עדיף על המודל הישן!")
                if st.button("🚀 Promote New Model"):
                    joblib.dump(trained_models[best_new_model], "models/best_model.joblib")
                    with open("assets/metrics.json","w") as f:
                        json.dump(metrics_comp, f)
                    st.success("✅ New model promoted as best model!")
            else:
                st.info("המודל הישן עדיין עדיף. לא עודכן Best Model.")

# --- Tab 7: Model History
with tab_hist:
    st.header("🕑 Model History")
    hist_file = "assets/model_history.csv"
    if os.path.exists(hist_file):
        hist_df = pd.read_csv(hist_file)
        st.dataframe(hist_df)
        if st.button("🔙 Rollback Last Model"):
            if len(hist_df)>1:
                prev_model = hist_df.iloc[-2]["model_path"]
                best_model = joblib.load(prev_model)
                st.session_state.best_model = best_model
                st.success("✅ Rolled back to previous model")
    else:
        st.info("No model history found yet.")

# --- Tab 8: Explain
with tab_explain:
    st.header("🧠 Model Explainability")

    try:
        # 🔹 Feature Importance (למודלים שתומכים בזה)
        if hasattr(best_model, "feature_importances_"):
            st.subheader("🔹 Feature Importance")
            fi = pd.DataFrame({
                "Feature": X.columns,
                "Importance": best_model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

        # 🔹 SHAP (אם נתמך)
        st.subheader("🔹 SHAP Summary Plot")
        import shap
        explainer = shap.Explainer(best_model, X)
        shap_values = explainer(X)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP not available: {e}")

        # ✅ Fallback: קורלציה פשוטה עם המטרה
        st.subheader("🔹 Correlation with Target (Fallback)")
        corr = df.corr()["status"].drop("status").sort_values(ascending=False)
        corr_df = corr.reset_index().rename(columns={"index": "Feature", "status": "Correlation"})
        fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h", title="Correlation with Target")
        st.plotly_chart(fig, use_container_width=True)
        tab_explain = st.tabs(["🧠 Explainability"])[0]
        with tab_explain:
            st.header("🧠 Model Explainability")
            try:
                if hasattr(best_model, "feature_importances_"):
                    importances = best_model.feature_importances_
                    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
                    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
                    
                    fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Top Feature Importances")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.warning(f"Explainability not available: {e}")


# --- Tab 9: About
with tab_about:
    st.header("ℹ️ About This App")
    st.markdown("""
    ### Parkinson’s ML App
    🧠 מערכת חכמה לניבוי מחלת פרקינסון על בסיס דאטה רפואי.

    **יכולות מרכזיות:**
    - ניתוח EDA אינטראקטיבי  
    - דאשבורד להשוואת מודלים  
    - ניבוי ידני או על קבצים  
    - אימון מחדש והשוואה למודל קיים  
    - היסטוריית אימונים + אפשרות Rollback  
    - ייצוא דוחות (Excel/PDF)

    👨‍💻 פותח במסגרת פרויקט סיום קורס ML & AI ב־Afeka.
    """)

    if st.button("📄 Download PDF Report"):
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.drawString(100, 750, "Parkinson’s ML Report")
        c.drawString(100, 730, f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")
        c.save()
        pdf_buffer.seek(0)
        st.download_button("Download PDF", data=pdf_buffer, file_name="report.pdf", mime="application/pdf")
        tab_about = st.tabs(["ℹ️ About"])[0]
with tab_about:
    st.header("ℹ️ About This Project")
    st.markdown("""
    🧠 **Parkinson’s ML App**  
    Final project for ML & AI course.  

    - Built with Streamlit, Scikit-learn, XGBoost, CatBoost, LightGBM.  
    - Provides full pipeline: EDA, model comparison, prediction, retraining.  
    - Supports multilingual interface + custom themes.  
    - Includes demo mode and model promotion system.  

    👨‍💻 Developed by: *Your Name*  
    """)


