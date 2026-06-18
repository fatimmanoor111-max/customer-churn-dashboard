# ============================================
# newtask.py — Customer Churn Dashboard
# ============================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir,
                "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        st.error(f"❌ CSV file nahi mili! Yahan rakho: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median())
    df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
    return df

df = load_data()

# ---- FEATURE ENGINEERING ----
df['total_services'] = (
    (df['OnlineSecurity'] == 'Yes').astype(int) +
    (df['OnlineBackup'] == 'Yes').astype(int) +
    (df['DeviceProtection'] == 'Yes').astype(int) +
    (df['TechSupport'] == 'Yes').astype(int) +
    (df['StreamingTV'] == 'Yes').astype(int) +
    (df['StreamingMovies'] == 'Yes').astype(int)
)

def segment(row):
    score = 0
    if row['MonthlyCharges'] > 70: score += 2
    if row['tenure'] > 36:         score += 2
    if row['total_services'] >= 4: score += 1
    if score >= 4:   return "High Value"
    elif score >= 2: return "Medium Value"
    return "Low Value"

df['Segment'] = df.apply(segment, axis=1)

# ---- ML MODELS ----
@st.cache_resource
def train_models(df):
    df_model = df.copy()
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        if col != 'customerID':
            df_model[col] = le.fit_transform(
                df_model[col].astype(str))

    X = df_model.drop(['customerID', 'Churn_Binary'], axis=1)
    y = df_model['Churn_Binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    return (X, X_test, y_test,
            lr, lr_pred, lr_prob,
            rf, rf_pred, rf_prob,
            scaler)

X, X_test, y_test, lr, lr_pred, lr_prob, \
rf, rf_pred, rf_prob, scaler = train_models(df)

# ---- SIDEBAR ----
st.sidebar.title("📊 Churn Dashboard")
st.sidebar.markdown("---")
section = st.sidebar.selectbox(
    "Select Section",
    [
        "🏠 Dataset Overview",
        "📈 EDA",
        "🔧 Feature Engineering",
        "👥 Customer Segmentation",
        "🤖 ML Models",
        "🔮 Churn Predictor",
        "💡 Business Insights"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Teyzix Core — DA-INT-1**")

# ============================================
# SECTIONS
# ============================================

# ---- Dataset Overview ----
if section == "🏠 Dataset Overview":
    st.title("📋 Dataset Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned", df['Churn_Binary'].sum())
    col3.metric("Churn Rate",
                f"{df['Churn_Binary'].mean()*100:.1f}%")
    col4.metric("Features", df.shape[1])

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        st.success("✅ No missing values!")
    else:
        st.write(missing)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ---- EDA ----
elif section == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")

    st.subheader("Churn Distribution & Tenure")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    churn_counts = df['Churn'].value_counts()
    axes[0].pie(churn_counts,
                labels=['No Churn', 'Churned'],
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                startangle=90)
    axes[0].set_title('Churn Distribution')

    axes[1].hist(df[df['Churn']=='No']['tenure'],
                 bins=30, alpha=0.7,
                 color='#2ecc71', label='No Churn')
    axes[1].hist(df[df['Churn']=='Yes']['tenure'],
                 bins=30, alpha=0.7,
                 color='#e74c3c', label='Churned')
    axes[1].set_title('Tenure by Churn')
    axes[1].set_xlabel('Tenure (months)')
    axes[1].legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Churn by Category")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    contract_churn = (df.groupby('Contract')['Churn_Binary']
                      .mean() * 100)
    axes2[0].bar(contract_churn.index,
                 contract_churn.values,
                 color=['#3498db','#e67e22','#9b59b6'])
    axes2[0].set_title('Churn Rate by Contract (%)')
    axes2[0].tick_params(axis='x', rotation=20)

    internet_churn = (df.groupby('InternetService')['Churn_Binary']
                      .mean() * 100)
    axes2[1].bar(internet_churn.index,
                 internet_churn.values,
                 color=['#1abc9c','#e74c3c','#f39c12'])
    axes2[1].set_title('Churn Rate by Internet (%)')
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    num_df = df[['tenure','MonthlyCharges',
                 'TotalCharges','Churn_Binary']]
    sns.heatmap(num_df.corr(), annot=True,
                cmap='RdYlGn', fmt='.2f',
                ax=ax3, linewidths=0.5)
    st.pyplot(fig3)
    plt.close()

# ---- Feature Engineering ----
elif section == "🔧 Feature Engineering":
    st.title("🔧 Feature Engineering")
    st.markdown("---")

    st.subheader("Engineered Features")
    st.dataframe(
        df[['customerID','tenure','MonthlyCharges',
            'total_services','Segment']].head(20),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Total Services Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    df['total_services'].value_counts().sort_index().plot(
        kind='bar', ax=ax, color='#3498db')
    ax.set_xlabel('Number of Services')
    ax.set_ylabel('Count')
    ax.set_title('Services Count Distribution')
    st.pyplot(fig)
    plt.close()

# ---- Customer Segmentation ----
elif section == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 High Value",
                (df['Segment']=='High Value').sum())
    col2.metric("📊 Medium Value",
                (df['Segment']=='Medium Value').sum())
    col3.metric("📉 Low Value",
                (df['Segment']=='Low Value').sum())

    st.markdown("---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seg_counts = df['Segment'].value_counts()
    axes[0].pie(seg_counts,
                labels=seg_counts.index,
                autopct='%1.1f%%',
                colors=['#f1c40f','#3498db','#e74c3c'])
    axes[0].set_title('Segment Distribution')

    seg_churn = (df.groupby('Segment')['Churn_Binary']
                 .mean() * 100)
    bars = axes[1].bar(seg_churn.index,
                       seg_churn.values,
                       color=['#e74c3c','#f39c12','#2ecc71'])
    axes[1].set_title('Churn Rate by Segment (%)')
    for bar, val in zip(bars, seg_churn.values):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontweight='bold')
    st.pyplot(fig)
    plt.close()

    st.subheader("Segment Details")
    st.dataframe(
        df[['customerID','Segment',
            'MonthlyCharges','tenure']].head(20),
        use_container_width=True
    )

# ---- ML Models ----
elif section == "🤖 ML Models":
    st.title("🤖 ML Model Performance")
    st.markdown("---")

    results = pd.DataFrame({
        "Metric": ["Accuracy","Precision",
                   "Recall","F1 Score","ROC-AUC"],
        "Logistic Regression": [
            round(accuracy_score(y_test, lr_pred), 4),
            round(precision_score(y_test, lr_pred), 4),
            round(recall_score(y_test, lr_pred), 4),
            round(f1_score(y_test, lr_pred), 4),
            round(roc_auc_score(y_test, lr_prob), 4)
        ],
        "Random Forest": [
            round(accuracy_score(y_test, rf_pred), 4),
            round(precision_score(y_test, rf_pred), 4),
            round(recall_score(y_test, rf_pred), 4),
            round(f1_score(y_test, rf_pred), 4),
            round(roc_auc_score(y_test, rf_prob), 4)
        ]
    })

    st.dataframe(results.set_index("Metric"),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results['Metric']))
    width = 0.35
    ax.bar(x - width/2,
           results['Logistic Regression'],
           width, label='Logistic Regression',
           color='#3498db')
    ax.bar(x + width/2,
           results['Random Forest'],
           width, label='Random Forest',
           color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(results['Metric'])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title('Model Performance Comparison')
    st.pyplot(fig)
    plt.close()

# ---- Churn Predictor ----
elif section == "🔮 Churn Predictor":
    st.title("🔮 Churn Risk Predictor")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        tenure  = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)",
                                   0.0, 200.0, 70.0)
        total   = st.number_input("Total Charges ($)",
                                   0.0, 10000.0, 1000.0)
    with col2:
        contract = st.selectbox("Contract Type",
            ["Month-to-month","One year","Two year"])
        internet = st.selectbox("Internet Service",
            ["Fiber optic","DSL","No"])
        payment  = st.selectbox("Payment Method",
            ["Electronic check","Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])

    if st.button("🔍 Predict Churn Risk",
                 use_container_width=True):
        avg_row = X.mean().to_frame().T
        avg_row["tenure"]         = tenure
        avg_row["MonthlyCharges"] = monthly
        avg_row["TotalCharges"]   = total

        prob = rf.predict_proba(avg_row)[0][1]

        st.markdown("---")
        st.metric("Churn Probability", f"{prob:.1%}")

        if prob >= 0.60:
            st.error("🔴 HIGH RISK — Immediate action needed!")
        elif prob >= 0.35:
            st.warning("🟡 MEDIUM RISK — Monitor this customer")
        else:
            st.success("🟢 LOW RISK — Customer is stable")

# ---- Business Insights ----
elif section == "💡 Business Insights":
    st.title("💡 Business Insights Report")
    st.markdown("---")

    churned      = df['Churn_Binary'].sum()
    churn_rate   = df['Churn_Binary'].mean() * 100
    avg_charge   = df[df['Churn_Binary']==1]['MonthlyCharges'].mean()
    revenue_loss = churned * avg_charge

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned Customers", churned)
    col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Monthly Revenue Loss", f"${revenue_loss:,.0f}")

    st.markdown("---")
    st.subheader("🔍 Key Findings")
    st.success("""
    • Month-to-month customers churn more.
    • Electronic check users have higher churn.
    • Low-tenure customers are high risk.
    • Fiber internet users churn more.
    """)

    st.markdown("---")
    st.subheader("📊 Churn Rate by Category")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    contract_churn = (df.groupby('Contract')['Churn_Binary']
                      .mean() * 100)
    bars1 = axes[0,0].bar(contract_churn.index,
                           contract_churn.values,
                           color=['#2ecc71','#f39c12','#e74c3c'])
    axes[0,0].set_title('Churn Rate by Contract Type',
                         fontweight='bold')
    axes[0,0].set_ylabel('Churn Rate (%)')
    axes[0,0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars1, contract_churn.values):
        axes[0,0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontweight='bold')

    pay_churn = (df.groupby('PaymentMethod')['Churn_Binary']
                 .mean() * 100)
    axes[0,1].barh(pay_churn.index,
                   pay_churn.values, color='#3498db')
    axes[0,1].set_title('Churn Rate by Payment Method',
                         fontweight='bold')
    axes[0,1].set_xlabel('Churn Rate (%)')

    internet_churn = (df.groupby('InternetService')['Churn_Binary']
                      .mean() * 100)
    bars3 = axes[1,0].bar(internet_churn.index,
                           internet_churn.values,
                           color=['#1abc9c','#e74c3c','#f39c12'])
    axes[1,0].set_title('Churn Rate by Internet Service',
                         fontweight='bold')
    axes[1,0].set_ylabel('Churn Rate (%)')
    for bar, val in zip(bars3, internet_churn.values):
        axes[1,0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontweight='bold')

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0,12,24,48,72],
        labels=['0-1yr','1-2yr','2-4yr','4+yr'])
    tenure_churn = (df.groupby('tenure_group',
                               observed=True)['Churn_Binary']
                    .mean() * 100)
    bars4 = axes[1,1].bar(
        tenure_churn.index.astype(str),
        tenure_churn.values,
        color=['#e74c3c','#e67e22','#f1c40f','#2ecc71'])
    axes[1,1].set_title('Churn Rate by Tenure Group',
                         fontweight='bold')
    axes[1,1].set_ylabel('Churn Rate (%)')
    for bar, val in zip(bars4, tenure_churn.values):
        axes[1,1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚠️ High Risk Traits")
        st.error("""
        🔴 Danger Signs:
        - Month-to-month contract
        - Electronic check payment
        - Fiber optic internet
        - Tenure less than 12 months
        - 0 to 1 services only
        """)
    with col2:
        st.subheader("✅ Safe Customer Signs")
        st.success("""
        🟢 Safe Signs:
        - Two year contract
        - Auto payment method
        - Tenure more than 36 months
        - 4+ services subscribed
        - DSL or No internet
        """)

    st.markdown("---")
    st.subheader("💡 Recommendations")
    st.info("""
    1. Promote annual contracts with discount offers
    2. Auto-payment incentives to reduce electronic check usage
    3. Loyalty rewards for fiber optic customers
    4. Onboarding support for first 6 months customers
    5. Bundle services to increase service count
    """)
