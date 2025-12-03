# home_page.py
import streamlit as st


def render_home():
    st.markdown("### Choose a model to get started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-header">
                <span class="icon">🧮</span>
                <span>General Maternal Model</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            """
This model focuses on **overall maternal health**, using information such as:

- Blood pressure, blood sugar, BMI  
- Previous complications and diabetes  
- Mental health and heart rate  

Use this for **general screening** and broader maternal risk assessment.
            """
        )
        if st.button("Use General Maternal Model", use_container_width=True):
            st.session_state["page"] = "General"
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-header">
                <span class="icon">🩺</span>
                <span>Pregnancy / Antenatal Model</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            """
This model is **more detailed for pregnancy**, using:

- Gestational age, weight, height  
- Tetanus doses and blood pressure  
- Infection screening (VDRL, HBsAg)  

Use this in **antenatal clinic settings** for visit-based risk assessment.
            """
        )
        if st.button("Use Pregnancy / Antenatal Model", use_container_width=True):
            st.session_state["page"] = "Pregnancy"
        st.markdown("</div>", unsafe_allow_html=True)
