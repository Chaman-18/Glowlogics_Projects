import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="FinTrack | Loan Underwriting", layout="wide")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        with open('loan_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

assets = load_assets()

if assets is None:
    st.error("⚠️ 'loan_model.pkl' not found. Please ensure the training script has been run.")
    st.stop()

model = assets['model']

# --- HEADER ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🏦 Chaman Loan Approval")
    st.markdown("Automated Credit Risk Assessment & Asset Verification")
with header_col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)

st.divider()

# --- INPUT SECTION (3-Column Layout) ---
st.subheader("📋 Applicant Information")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### 👤 Personal Details")
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5])
    education = st.segmented_control("Education Status", ["Graduate", "Not Graduate"], default="Graduate")
    employed = st.segmented_control("Self Employed?", ["Yes", "No"], default="No")

with col_b:
    st.markdown("### 💰 Financials")
    income = st.number_input("Annual Income (₹)", min_value=10000, value=500000)
    loan_amt = st.number_input("Requested Loan Amount (₹)", min_value=10000, value=200000)
    term = st.slider("Loan Term (Years)", 2, 20, 10)

with col_c:
    st.markdown("### 📊 Credit Score")
    cibil = st.select_slider("CIBIL Score", options=list(range(300, 901)), value=700)
    st.caption("Standard range: 300 - 900")

st.markdown("### 🏗️ Asset Declaration")
a1, a2, a3, a4 = st.columns(4)
res_asset = a1.number_input("Residential (₹)", value=100000)
com_asset = a2.number_input("Commercial (₹)", value=50000)
lux_asset = a3.number_input("Luxury (₹)", value=20000)
bank_asset = a4.number_input("Bank (₹)", value=150000)

st.divider()

# --- PREDICTION ---
if st.button("🔥 GENERATE RISK REPORT"):
    # Encoding
    edu_val = 0 if education == "Graduate" else 1
    emp_val = 1 if employed == "Yes" else 0
    
    input_data = np.array([[
        dependents, edu_val, emp_val, income, loan_amt, term, cibil, 
        res_asset, com_asset, lux_asset, bank_asset
    ]])
    
    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]
    conf_score = probs[0] * 100 if prediction[0] == 0 else probs[1] * 100
    risk_percent = probs[1] * 100

    # Results Section
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction[0] == 0:
            st.success("### ✅ APPROVED")
            st.metric("Approval Confidence", f"{conf_score:.1f}%")
        else:
            st.error("### ❌ REJECTED")
            st.metric("Rejection Risk", f"{risk_percent:.1f}%")
        
        # Risk Tag
        if risk_percent < 30:
            st.info("Tier: **Prime (Low Risk)**")
        elif 30 <= risk_percent < 70:
            st.warning("Tier: **Sub-Prime (Mid Risk)**")
        else:
            st.error("Tier: **High Risk**")

    with res_col2:
        # Comparison Chart: Income vs Loan
        comparison_df = pd.DataFrame({
            "Category": ["Annual Income", "Loan Requested"],
            "Amount": [income, loan_amt]
        })
        fig = px.bar(comparison_df, x="Category", y="Amount", color="Category", 
                     title="Income vs Loan Ratio", color_discrete_sequence=["#2ecc71", "#3498db"])
        st.plotly_chart(fig, use_container_width=True)

    # Detailed Analysis Tabs
    tab1, tab2 = st.tabs(["Asset Breakdown", "Model Probability"])
    
    with tab1:
        asset_map = pd.DataFrame({
            'Type': ['Residential', 'Commercial', 'Luxury', 'Bank'],
            'Value': [res_asset, com_asset, lux_asset, bank_asset]
        })
        fig_donut = px.pie(asset_map, values='Value', names='Type', hole=0.5, title="Collateral Mix")
        st.plotly_chart(fig_donut)

    with tab2:
        fig_prob = go.Figure(go.Bar(
            x=['Approved', 'Rejected'],
            y=[probs[0], probs[1]],
            marker_color=['#2ecc71', '#e74c3c']
        ))
        fig_prob.update_layout(title="Raw Model Probability Distribution")
        st.plotly_chart(fig_prob)

st.markdown("---")
st.caption("INTERNAL BANKING USE ONLY • Version 2.0 • Data Source: loan_approval_dataset.csv")