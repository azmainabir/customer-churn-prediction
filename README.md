# 📊 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An end-to-end machine learning system that predicts whether a telecom customer will churn — built with XGBoost, SHAP explainability, and a live Streamlit dashboard.

---

## 🔴 Live Demo

👉 **[View Live App](#)** ← *(update after deployment)*

---

## ✨ Features

- 🤖 **3 ML Models** — Logistic Regression, Random Forest, XGBoost compared side by side
- 🔍 **SHAP Explainability** — See exactly why the model made each prediction
- 📊 **ROC Curve** — Visual model performance comparison
- 🔥 **Confusion Matrix** — Heatmap of prediction accuracy
- 📈 **Feature Importance** — Top 15 features driving churn
- 🎛️ **Interactive Sidebar** — Enter any customer profile and get instant prediction
- 🟢 **Risk Level Indicator** — Low / Medium / High risk classification
- 💡 **Real Dataset** — Trained on 7,000+ real telecom customers

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core programming language |
| Streamlit | Interactive web dashboard |
| XGBoost | Primary prediction model |
| scikit-learn | ML pipeline & metrics |
| SHAP | Model explainability |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Chart generation |
| Joblib | Model serialization |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                  ← Streamlit web app
├── train_model.py          ← Model training script
├── requirements.txt        ← Python dependencies
│
├── data/
│   └── telco_churn.csv     ← Dataset (7,043 customers)
│
├── models/
│   ├── xgboost_model.pkl   ← Trained XGBoost model
│   └── feature_names.pkl   ← Feature list
│
├── assets/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── churn_distribution.png
│
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/azmainabir/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Open in browser

```
http://localhost:8501
```

---

## 📊 Model Performance

| Model | AUC Score |
|-------|-----------|
| Logistic Regression | 0.8346 |
| Random Forest | 0.8113 |
| XGBoost | 0.8044 |

> **Key Insight:** Logistic Regression performed best on this dataset, showing that simpler models can outperform complex ones on structured tabular data. XGBoost is retained for its superior SHAP explainability and feature interaction analysis.

---

## 📦 Dataset

- **Source:** IBM Telco Customer Churn Dataset (via Kaggle)
- **Size:** 7,043 customers, 21 features
- **Churn Rate:** 26.6%
- **Features:** Contract type, tenure, monthly charges, internet service, tech support, and more

---

## 🗺️ Roadmap

- [x] Data cleaning and preprocessing
- [x] Train 3 ML models and compare
- [x] SHAP explainability
- [x] Streamlit dashboard
- [x] Deploy to Streamlit Cloud
- [ ] Add LIME explainability
- [ ] Add batch prediction (upload CSV)
- [ ] Add email alert for high-risk customers

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Developer

**Azmain Tahmid Abir** — CSE Student @ Daffodil International University · Passionate about Data Science · AI Engineering · Cyber Security · Software Development.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/azmain-abir)
[![GitHub](https://img.shields.io/badge/GitHub-azmainabir-181717?style=flat-square&logo=github)](https://github.com/azmainabir)
