import streamlit as st
import numpy as np

from utils import (
    load_models,
    FEATURES_DS2,
    get_shap_values,
    plot_shap_bar,
    plot_shap_waterfall,
    create_pdf_report,
    format_risk_label,
)


# -------- Risk color helper (same as general page) --------
def risk_color(label: str):
    label = label.lower()
    if "low" in label:
        return "#2ECC71"   # green
    if "high" in label:
        return "#E74C3C"   # red
    return "#F5B041"       # moderate / other (orange)


def render_pregnancy_model():
    # load models (we use model_ds2 here)
    model_ds2, model_ds3 = load_models()

    st.header("🩺 Pregnancy / Antenatal Model")
    st.markdown(
        "<p class='section-caption'>Use this for pregnancy visits with gestational age, BP, infection screening and other antenatal data.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Enter pregnancy and clinical information")

    # ---------------- INPUT COLUMNS (symmetry) ----------------
    col_left, col_right = st.columns([1.4, 1.0])

    # LEFT COLUMN → Numeric features
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### Pregnancy details & vitals", unsafe_allow_html=False)
        age = st.number_input("Age (years)", min_value=10, max_value=60, value=25, step=1)
        tt_doses = st.number_input("TT doses received", min_value=0, max_value=5, value=2, step=1)
        gest_age = st.number_input("Gestational age (weeks)", min_value=4, max_value=42, value=20, step=1)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.5)
        systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=110, step=1)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=70, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN → Infection markers (card with padding)
    with col_right:
        st.markdown('<div class="card" style="padding-left:1.4rem; padding-right:1.4rem;">', unsafe_allow_html=True)
        st.markdown("##### Infection screening", unsafe_allow_html=False)
        vdrl = st.radio(
            "VDRL (Syphilis test)",
            ["Negative", "Positive"],
            horizontal=True,
            index=0,
        )
        hbsag = st.radio(
            "HBsAg (Hepatitis B)",
            ["Negative", "Positive"],
            horizontal=True,
            index=0,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================================================
    #                         ACTION BUTTONS
    # ===============================================================
    st.markdown("")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict_clicked = st.button("🔮 Predict & Explain", key="predict_pregnancy", use_container_width=True)

    with col_btn2:
        home_clicked = st.button("🏠 Back to Home", key="home_pregnancy", use_container_width=True)

    if home_clicked:
        st.session_state["page"] = "Home"
        return

    if not predict_clicked:
        return

    # ===============================================================
    #                        PREPARE INPUT VECTOR
    # ===============================================================
    bin_map = {
        "Negative": 0,
        "Positive": 1,
    }

    input_data = {
        "Age": age,
        "TT_Doses": tt_doses,
        "Gestational_Age": gest_age,
        "Weight": weight,
        "VDRL": bin_map[vdrl],
        "HBsAg": bin_map[hbsag],
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
    }

    x = np.array([[input_data[f] for f in FEATURES_DS2]], dtype=float)

    # ===============================================================
    #                        MODEL PREDICTION
    # ===============================================================
    pred = model_ds2.predict(x)[0]
    classes = getattr(model_ds2, "classes_", None)
    proba = model_ds2.predict_proba(x)[0] if hasattr(model_ds2, "predict_proba") else None

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

    # ===============================================================
    #                       DISPLAY PREDICTION
    # ===============================================================
    st.markdown("### 🧾 Prediction")

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns([1.4, 1.1, 1.3])

    with col_p1:
        st.markdown("<div class='result-title'>Pregnancy-related risk estimation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <span class="risk-badge {badge_class}">{nice_label}</span>
            <div class="risk-main-value" style="color:{color}; margin-top:0.6rem;">
                {nice_label}
            </div>
            <div class="risk-subtext">
                Based on gestational age, blood pressure, weight and infection markers.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        if proba is not None:
            idx = int(pred)
            conf = float(proba[idx])
            st.markdown("**Model confidence**")
            st.markdown(f"**{conf*100:.1f}%**")
            st.progress(conf)
        else:
            st.markdown("**Model confidence**")
            st.write("_Confidence not available for this model._")

    with col_p3:
        st.markdown("**Quick interpretation**")
        if "low" in nice_label.lower():
            st.write(
                "The model suggests a **low pregnancy-related risk** for this visit. "
                "Maintain routine antenatal care and lifestyle advice."
            )
        elif "high" in nice_label.lower():
            st.write(
                "The model suggests a **high pregnancy-related risk**. "
                "Consider urgent clinical review, further investigations or referral."
            )
        else:
            st.write(
                "The model suggests a **moderate pregnancy-related risk**. "
                "Monitor blood pressure, weight and infection markers closely."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if proba is not None and classes is not None:
        with st.expander("🔍 See full probability distribution"):
            for c, p in zip(classes, proba):
                st.write(f"- {format_risk_label(c)}: `{p:.3f}`")

    # ===============================================================
    #                       XAI (SHAP)
    # ===============================================================
    st.markdown("### 🧠 Why did the model say this?")
    st.markdown(
        "<p class='section-caption'>The following plots highlight which antenatal features most influenced this prediction.</p>",
        unsafe_allow_html=True,
    )

    shap_values, base_value = get_shap_values(
        model_ds2,
        x,
        predicted_class_index=int(pred) if classes is not None else None,
    )

    tab_bar, tab_waterfall = st.tabs(["Bar Plot (feature impact)", "Waterfall Plot (step-by-step)"])

    with tab_bar:
        st.markdown("<div class='shap-card'>", unsafe_allow_html=True)
        st.pyplot(
            plot_shap_bar(
                shap_values,
                FEATURES_DS2,
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
                FEATURES_DS2,
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
    st.write("**Top contributing factors for this pregnancy visit:**")
    for i in top3:
        direction = "raised the risk" if shap_values[i] > 0 else "lowered the risk"
        st.write(f"- **{FEATURES_DS2[i]}** → {direction}")

    # ===============================================================
    #                       PDF REPORT
    # ===============================================================
    st.markdown("### 📄 Download report")

    top5 = idx_sorted[:5]
    top_contribs = [(FEATURES_DS2[i], float(shap_values[i])) for i in top5]

    proba_dict = (
        {str(c): float(p) for c, p in zip(classes, proba)}
        if (proba is not None and classes is not None)
        else None
    )

    pdf_buffer = create_pdf_report(
        model_name="Pregnancy / Antenatal Model",
        input_dict=input_data,
        pred_label=nice_label,
        proba_dict=proba_dict,
        shap_contribs=top_contribs,
    )

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="maternal_risk_report_pregnancy.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
