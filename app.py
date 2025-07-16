import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# 🌟 Header
st.markdown("""
    <div style='text-align: center; padding: 10px'>
        <h1 style='color: #4A90E2;'>📊 Customer Churn Prediction App</h1>
        <h4 style='color: grey;'>Predict whether a customer is likely to churn using machine learning</h4>
        <hr style="border:1px solid #ddd;">
    </div>
""", unsafe_allow_html=True)

# 📌 Model training (simple RF for demo)
@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop(['customerID'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2)
    model.fit(X_res, y_res)

    return model, X.columns

model, feature_cols = train_model()

# 📁 Upload CSV
st.markdown("### 📁 Upload Customer Data (CSV format)")
uploaded_file = st.file_uploader("Upload your customer dataset", type="csv")

# 🚀 Prediction Logic
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    input_data = pd.get_dummies(data, drop_first=True)

    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_cols]  # Ensure column order

    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)[:, 1]

    data['Churn_Predicted'] = np.where(preds == 1, 'Yes', 'No')
    data['Churn_Probability (%)'] = np.round(probs * 100, 2)

    # 🎉 Summary Metrics
    churn_count = np.sum(preds)
    total = len(preds)
    churn_percent = churn_count / total * 100

    st.markdown(f"""
    <div style="background-color:#e8f4fc; padding:15px; border-radius:10px; margin-top:20px;">
        <h4>🔎 Prediction Summary</h4>
        <p><strong>Total Customers:</strong> {total}</p>
        <p><strong>Predicted to Churn:</strong> {churn_count} ({churn_percent:.2f}%)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📄 Prediction Results")
    st.dataframe(data[['Churn_Predicted', 'Churn_Probability (%)'] + list(data.columns.difference(['Churn_Predicted', 'Churn_Probability (%)']))])

    st.download_button(
        label="📥 Download Result as CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file to begin predictions.")

# 🔚 Footer
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <small>🧠 Built with ❤️ using Streamlit & Random Forest</small>
    </div>
""", unsafe_allow_html=True)
import plotly.express as px

st.markdown("### 📊 Churn Prediction Overview")

# ✅ Calculate churn summary
churn_count = data['Churn_Predicted'].value_counts().get('Yes', 0)
total = len(data)

# 📈 Pie chart of churn vs no churn
pie_data = pd.DataFrame({'Churn': ['Yes', 'No'], 'Count': [churn_count, total - churn_count]})
fig_pie = px.pie(pie_data, names='Churn', values='Count', title='📈 Churn Distribution', color_discrete_sequence=['red', 'green'])
st.plotly_chart(fig_pie, use_container_width=True)

# 📊 Histogram of churn probabilities
fig_hist = px.histogram(data, x='Churn_Probability (%)', nbins=20, title='🔢 Churn Probability Distribution', color='Churn_Predicted')
st.plotly_chart(fig_hist, use_container_width=True)
