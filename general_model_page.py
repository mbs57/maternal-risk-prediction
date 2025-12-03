import streamlit as st
import numpy as np

from utils import (
    load_models,
    FEATURES_DS3,
    get_shap_values,
    plot_shap_bar,
    plot_shap_waterfall,
    create_pdf_report,
    format_risk_label,
)


def risk_color(label: str):
    label = label.lower()
    if "low" in label:
        return "#2ECC71"   # green
    if "high" in label:
        return "#E74C3C"   # red
    return "#F5B041"       # moderate (orange)


def render_general_model():
    model_ds2, model_ds3 = load_models()

    st.header("🧮 General Maternal Model")
    st.markdown(
        "<p class='section-caption'>Provide basic vital signs and clinical history to estimate maternal risk.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Enter patient information")

    # ---------------- INPUT COLUMNS (symmetry) ----------------
    col_left, col_right = st.columns([1.2, 1.0])

    # LEFT: vitals
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### Pregnancy details & vitals", unsafe_allow_html=False)
        age = st.number_input("Age (years)", min_value=10, max_value=60, value=25, step=1)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80, step=1)
        bs = st.number_input("Blood Sugar (BS)", min_value=40, max_value=400, value=100, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        heart_rate = st.number_input("Heart rate (bpm)", min_value=40, max_value=200, value=80, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: clinical history
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### Clinical history", unsafe_allow_html=False)
        prev_comp = st.radio("Previous complications", ["No", "Yes"], horizontal=True)
        pre_diab = st.radio("Preexisting diabetes", ["No", "Yes"], horizontal=True)
        gest_diab = st.radio("Gestational diabetes", ["No", "Yes"], horizontal=True)
        mental_health = st.radio("Mental health issue", ["No", "Yes"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ACTION BUTTONS ----------------
    st.markdown("")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_clicked = st.button("🔮 Predict & Explain", key="predict_general", use_container_width=True)
    with col_btn2:
        home_clicked = st.button("🏠 Back to Home", key="home_general", use_container_width=True)

    if home_clicked:
        st.session_state["page"] = "Home"
        return
    if not predict_clicked:
        return

    # ---------------- PREPARE INPUTS ----------------
    yn = {"No": 0, "Yes": 1}
    input_data = {
        "Age": age,
        "Diastolic": diastolic,
        "BS": bs,
        "BMI": bmi,
        "Previous Complications": yn[prev_comp],
        "Preexisting Diabetes": yn[pre_diab],
        "Gestational Diabetes": yn[gest_diab],
        "Mental Health": yn[mental_health],
        "Heart Rate": heart_rate,
    }
    x = np.array([[input_data[f] for f in FEATURES_DS3]], dtype=float)

    # ---------------- PREDICTION ----------------
    pred = model_ds3.predict(x)[0]
    classes = getattr(model_ds3, "classes_", None)
    proba = model_ds3.predict_proba(x)[0] if hasattr(model_ds3, "predict_proba") else None

    if classes is not None:
        raw_label = classes[int(pred)]
    else:
        raw_label = pred

    nice_label = format_risk_label(raw_label)
    color = risk_color(nice_label)

    badge_class = "risk-moderate"
    if "low" in nice_label.lower():
        badge_class = "risk-low"
    elif "high" in nice_label.lower():
        badge_class = "risk-high"

    st.markdown("### 🧾 Prediction")

    # --------- Result card layout ----------
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns([1.4, 1.1, 1.2])

    with col_r1:
        st.markdown("<div class='result-title'>Overall risk estimation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <span class="risk-badge {badge_class}">{nice_label}</span>
            <div class="risk-main-value" style="color:{color}; margin-top:0.6rem;">
                {nice_label}
            </div>
            <div class="risk-subtext">
                This reflects the model's assessment based on the provided vitals and history.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_r2:
        if proba is not None:
            idx = int(pred)
            conf = float(proba[idx])
            st.markdown("**Model confidence**")
            st.markdown(f"**{conf*100:.1f}%**")
            st.progress(conf)
        else:
            st.markdown("**Model confidence**")
            st.write("_Confidence not available for this model._")

    with col_r3:
        st.markdown("**Quick interpretation**")
        if "low" in nice_label.lower():
            st.write(
                "The model suggests a **low level of maternal risk** based on the current inputs. "
                "Continue routine monitoring and healthy lifestyle measures."
            )
        elif "high" in nice_label.lower():
            st.write(
                "The model suggests a **high level of maternal risk**. "
                "Consider closer clinical evaluation and follow-up."
            )
        else:
            st.write(
                "The model suggests a **moderate level of maternal risk**. "
                "Monitor closely and address modifiable risk factors where possible."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Show probability distribution in a small card
    if proba is not None and classes is not None:
        with st.expander("🔍 See full probability distribution"):
            for c, p in zip(classes, proba):
                st.write(f"- {format_risk_label(c)}: `{p:.3f}`")

    # ---------------- XAI (SHAP) ----------------
    st.markdown("### 🧠 Why did the model say this?")
    st.markdown(
        "<p class='section-caption'>These plots show which features pushed the prediction higher or lower.</p>",
        unsafe_allow_html=True,
    )

    shap_values, base_value = get_shap_values(
        model_ds3,
        x,
        predicted_class_index=int(pred) if classes is not None else None,
    )

    tab_bar, tab_waterfall = st.tabs(["Bar Plot (feature impact)", "Waterfall Plot (step-by-step)"])

    with tab_bar:
        st.markdown("<div class='shap-card'>", unsafe_allow_html=True)
        st.pyplot(
            plot_shap_bar(
                shap_values,
                FEATURES_DS3,
                "Feature impact on prediction",
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_waterfall:
        st.markdown("<div class='shap-card'>", unsafe_allow_html=True)
        st.pyplot(
            plot_shap_waterfall(
                shap_values,
                base_value,
                x[0],
                FEATURES_DS3,
                "How each feature shifts risk",
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### In simple terms")
    st.write(
        "Bars pushing **to the right** increase the estimated risk, while bars pushing "
        "**to the left** reduce it."
    )

    idx_sorted = np.argsort(np.abs(shap_values))[::-1]
    top3 = idx_sorted[:3]
    st.write("**Top contributing factors for this patient:**")
    for i in top3:
        direction = "raised the risk" if shap_values[i] > 0 else "lowered the risk"
        st.write(f"- **{FEATURES_DS3[i]}** → {direction}")

    # ---------------- PDF ----------------
    st.markdown("### 📄 Download report")

    top5 = idx_sorted[:5]
    top_contribs = [(FEATURES_DS3[i], float(shap_values[i])) for i in top5]

    proba_dict = (
        {str(c): float(p) for c, p in zip(classes, proba)}
        if (proba is not None and classes is not None)
        else None
    )

    pdf_buffer = create_pdf_report(
        model_name="General Maternal Model",
        input_dict=input_data,
        pred_label=nice_label,
        proba_dict=proba_dict,
        shap_contribs=top_contribs,
    )

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="maternal_risk_report_general.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
