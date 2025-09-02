# Parkinson's Prediction Project 🧠

## 📂 Structure
- app/streamlit_app.py → Streamlit UI
- app/model_pipeline.py → Training pipeline
- data/parkinsons.csv → Dataset (UCI Parkinson's)
- models/best_model.joblib → Saved model (after training)
- assets/ → Metrics + plots
- requirements.txt → Libraries
- README.md → Instructions

## ▶️ Run
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 📊 Tabs in the App
1. Data & EDA → Exploratory analysis, plots, stats
2. Models → Comparison of ML models + metrics
3. Prediction → Manual / CSV prediction
4. Train New Model → Retraining pipeline
