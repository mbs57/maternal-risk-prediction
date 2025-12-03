# app.py
import streamlit as st

from utils import apply_global_css
from home_page import render_home
from general_model_page import render_general_model
from pregnancy_model_page import render_pregnancy_model

st.set_page_config(
    page_title="Maternal Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

apply_global_css()

# ---------------- Session state ----------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# ---------------- Sidebar: user manual ----------------
st.sidebar.markdown(
    """
**User manual**

1. Start on the **Home** screen and choose one of the two models:  
   - *General Maternal Model* – for overall maternal health.  
   - *Pregnancy / Antenatal Model* – for clinic visits during pregnancy.  

2. Fill in the numeric inputs and toggles with the patient’s information.  

3. Click **“Predict & Explain”** to see:  
   - A risk label (e.g. *Low risk*, *High risk*).  
   - A confidence bar.  
   - Simple explanations of which features increased or decreased the risk.  

4. You can download a **PDF report** or click **Home** on a model page to return.

_Created by **Mrinal Basak Shuvo**._
"""
)

# ---------------- Title & welcome text ----------------
st.markdown(
    '<div class="main-title">Maternal Risk Prediction</div>',
    unsafe_allow_html=True,
)

if st.session_state["page"] == "Home":
    st.markdown(
        '<div class="welcome-text">'
        'Welcome! This tool offers two complementary machine learning models to help estimate maternal risk.'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------- Routing ----------------
page = st.session_state["page"]

if page == "Home":
    render_home()
elif page == "General":
    render_general_model()
elif page == "Pregnancy":
    render_pregnancy_model()
