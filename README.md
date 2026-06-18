TOPIC: Customer Behavior Analytics & Churn Prediction Dashboard

Teyzix Core Internship — Task ID: DA-INT-1

Domain: Data Analytics | Level: Intermediate
Submitted By: Noor Fatima
Submission Date: June 2026
_________________________________________________________________________________________
DESCRIPTION:
This project is an interactive Customer Churn Prediction Dashboard
built using Python and Streamlit. It analyses telecom customer data
to identify churn patterns, segment customers by value, train ML
models, and predict individual churn risk.

 ⚙️ Requirements
Python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost         (optional — for XGB model)
shap            (optional — for explainability bonus)
streamlit       (optional — for dashboard bonus)
```

### Install all at once:
1. Make sure Python 3.8+ is installed
2. Install required libraries:
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
3. Run the app:
   streamlit run Newtask.py

4. Open your browser at:
   http://localhost:8501

🚀 How to Run

### Option A — Jupyter Notebook (Primary Deliverable)
```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All).

### Option B — Streamlit Dashboard (Bonus)
```bash
streamlit run newtask.py
```
Opens an interactive dashboard at `http://localhost:8501`

---

 Key Results (Approximate — actual values generated at runtime)

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.60 | ~0.85 |
| Random Forest | ~79% | ~0.57 | ~0.83 |
| XGBoost | ~81% | ~0.61 | ~0.86 |

---

 Key Findings

- **26.5% overall churn rate** — significantly high for the telecom sector
- **Month-to-month contracts** have ~42% churn vs ~3% for two-year contracts
- **First 12 months** are the most critical — churn rate exceeds 47%
- **Electronic check users** churn at ~45%, vs ~16% for auto-payment
- **Fiber optic customers** pay more but churn more (~42%)
- **Monthly revenue loss** from churn: ~$139,000/month

DATASET:
--------
Source: IBM Telco Customer Churn Dataset
Rows: 7,043 customers | Features: 21

MODELS USED:
------------
- Logistic Regression
- Random Forest Classifier

================================================
👩‍💻 Author
Noor Fatima
https://www.linkedin.com/in/noor-fatima-a902b7389/
https://github.com/fatimmanoor111-max
