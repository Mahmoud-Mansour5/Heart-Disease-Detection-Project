import sys
import os
import collections

# 1. IMPORTANT: Patch for Experta & Python 3.10+ compatibility
if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping

# 2. IMPORTANT: Fix Folder Paths to allow cross-folder imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
from ml_model.predict import run_inference
from rule_based_system.expert_system import HeartDiseaseExpertSystem

# --- UI Configuration ---
st.set_page_config(
    page_title="Heart Disease AI Diagnostic System", 
    page_icon="❤️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR: PATIENT DATA ENTRY
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.markdown("## 📋 Patient Data Entry")
    st.caption("Please input the required medical parameters to run the dual-AI analysis.")
    st.divider()

    with st.form("patient_form"):
        st.markdown("** Demographics**")
        age = st.number_input("Age", 1, 100, 62)
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)", horizontal=True)
        
        st.markdown("**🩸 Vitals & Blood**")
        trestbps = st.slider("Resting BP (trestbps)", 80, 200, 160)
        chol = st.slider("Cholesterol (chol)", 100, 600, 280)
        fbs = st.radio("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", horizontal=True)
        
        st.markdown("**Heart Rate**")
        cp = st.selectbox("Chest Pain Type (cp) [0-3]", [0, 1, 2, 3])
        restecg = st.selectbox("Resting ECG (restecg) [0-2]", [0, 1, 2])
        thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 110)
        exang = st.radio("Exercise Angina (exang)", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", horizontal=True)
        
        st.markdown("**more details**")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 2.5, step=0.1)
        slope = st.selectbox("ST Slope (slope) [0-2]", [0, 1, 2])
        ca = st.selectbox("Major Vessels Blocked (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3], format_func=lambda x: f"{x} - " + ("Normal" if x == 1 else "Fixed Defect" if x == 2 else "Reversible Defect"))

        st.write("") # Spacer
        submitted = st.form_submit_button(" ANALYSIS Now", use_container_width=True)

# ==========================================
# MAIN PAGE: RESULTS DASHBOARD
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2c3e50;'> Heart Disease Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d; font-weight: normal;'>Mahmoud Mansour</h4>", unsafe_allow_html=True)
st.divider()

if not submitted:
    # Welcome Screen when the app is first opened
    col_img, col_text = st.columns([1, 3])
    with col_text:
        st.info("👈 **Awaiting Data:** Please enter the patient's medical details in the sidebar on the left and click **'RUN AI ANALYSIS'** to view the results.")
        st.markdown("""
        **How this system works:**
        * **Machine Learning:** Uses a trained Decision Tree model to find hidden patterns in the patient's data.
        * **Expert System:** Uses Experta to apply strict, human-defined medical logic (Symbolic AI).
        """)

else:
    # When the user clicks the submit button, this section runs
    patient_data = {
        "age": age, "sex": sex, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "cp": cp, "thal": thal
    }

    # 1. Run Machine Learning Model
    res_ml, prob_ml = run_inference(patient_data)
    
    # 2. Run Expert System
    engine = HeartDiseaseExpertSystem()
    res_expert = engine.declare_facts(patient_data)
    reasons = engine.risk_details.get("reasons", [])
    expert_score = engine.risk_details.get("score", 0)

    # --- Display Results ---
    res_col1, spacer, res_col2 = st.columns([1, 0.1, 1])

    # Left Column: Machine Learning
    with res_col1:
        st.markdown("### 🤖 Sub-symbolic AI (ML)")
        st.caption("Decision Tree Classifier Output")
        
        container = st.container(border=True)
        with container:
            if res_ml == 0:
                st.error("#### 🔴 DIAGNOSIS: HIGH RISK (Heart Disease)")
            else:
                st.success("#### 🟢 DIAGNOSIS: LOW RISK (Healthy)")
            
            prob_value = prob_ml[0] if res_ml == 0 else prob_ml[1]
            st.metric(label="Model Confidence Score", value=f"{prob_value * 100:.1f}%")
            st.progress(float(prob_value))

    # Right Column: Expert System
    with res_col2:
        st.markdown("### 📜 Symbolic AI (Rules)")
        st.caption("Experta Logic Engine Output")
        
        container2 = st.container(border=True)
        with container2:
            if "HIGH" in res_expert.upper():
                st.error(f"#### 🔴 DIAGNOSIS: {res_expert.upper()}")
            elif "MODERATE" in res_expert.upper():
                st.warning(f"#### 🟡 DIAGNOSIS: {res_expert.upper()}")
            else:
                st.success(f"#### 🟢 DIAGNOSIS: {res_expert.upper()}")
            
            st.metric(label="Total Medical Risk Points", value=expert_score)
            
            with st.expander("🔍 **View Medical Reasoning (Rules Triggered)**", expanded=False):
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.write("ℹ️ *No specific strong risk rules were triggered.*")