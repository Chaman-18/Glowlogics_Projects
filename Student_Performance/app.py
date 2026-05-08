import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="EduPredict AI | Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True) # Fixed the parameter name here

# 2. Load Model Assets
@st.cache_resource
def load_assets():
    try:
        with open('student_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

assets = load_assets()

if assets is None:
    st.error("⚠️ Model file not found! Please run 'python train_model.py' first.")
    st.stop()

model = assets['model']
mappings = assets['mappings']

# 3. Sidebar Header & Configuration
st.sidebar.title("🎛️ Settings")
st.sidebar.markdown("Modify student parameters below:")

with st.sidebar.expander("👤 Profile & Environment", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    parent_edu = st.selectbox("Parent Education", ["High School", "Bachelors", "Masters", "PhD"])
    internet = st.radio("Home Internet", ["Yes", "No"], horizontal=True)
    extracurricular = st.radio("Extracurriculars", ["Yes", "No"], horizontal=True)

with st.sidebar.expander("📚 Academic Performance", expanded=True):
    study_hours = st.slider("Weekly Study Hours", 10, 40, 25)
    attendance = st.slider("Attendance Rate (%)", 50, 100, 80)
    past_scores = st.slider("Previous Scores", 50, 100, 75)

# 4. Main Dashboard Header
st.title("🎓 Student Performance Analytics AI")
st.write("Real-time academic forecasting and risk classification using Random Forest.")
st.markdown("---")

# 5. Analysis Trigger
if st.sidebar.button("✨ GENERATE REPORT", use_container_width=True):
    # Process Inputs based on model mappings
    input_features = [
        mappings['Gender'][gender],
        study_hours,
        attendance,
        past_scores,
        mappings['Parental_Education_Level'][parent_edu],
        mappings['Internet_Access_at_Home'][internet],
        mappings['Extracurricular_Activities'][extracurricular]
    ]
    
    # Get Probability
    prediction_prob = model.predict_proba([input_features])[0]
    pass_likelihood = prediction_prob[1] * 100
    
    # --- ROW 1: METRIC CARDS ---
    m1, m2, m3, m4 = st.columns(4)
    
    status = "PASS" if pass_likelihood >= 50 else "FAIL"
    
    m1.metric("Predicted Status", status)
    m2.metric("Success Probability", f"{pass_likelihood:.1f}%")
    m3.metric("Engagement Level", f"{attendance}%")
    m4.metric("Study Load", f"{study_hours}h")

    st.markdown("---")

    # --- ROW 2: DETAILED INSIGHTS ---
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("🎯 Risk Assessment")
        if pass_likelihood > 80:
            st.success("### ⭐ Excellent Standing\nThis student is a high performer with very low risk of failure.")
        elif pass_likelihood > 50:
            st.warning("### ⚠️ Average Standing\nPerformance is satisfactory but requires monitoring to prevent a decline.")
        else:
            st.error("### 🛑 At Risk\nImmediate corrective action and teacher support recommended.")

        # Local Feature Importance
        st.subheader("💡 Key Impact Factors")
        importance = pd.DataFrame({
            'Feature': assets['features'],
            'Impact': model.feature_importances_
        }).sort_values(by='Impact', ascending=True)
        
        fig_bar = px.bar(importance, x='Impact', y='Feature', orientation='h',
                         color='Impact', color_continuous_scale='Blues',
                         template="simple_white")
        fig_bar.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("📈 Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pass_likelihood,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 50], 'color': '#ff4b4b'},
                    {'range': [50, 80], 'color': '#ffa500'},
                    {'range': [80, 100], 'color': '#2eb82e'}]
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

else:
    # Default State
    st.info("👈 Modify student parameters in the sidebar and click **GENERATE REPORT** to see analysis.")
    
    # Overview Columns
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### **System Objective**")
        st.write("Identify academic failure risks early to allow for data-driven interventions and improved educational outcomes.")
    with col_b:
        st.markdown("#### **Algorithm**")
        st.write("Using an optimized **Random Forest Classifier** to analyze complex non-linear relationships between socioeconomic factors and grades.")
