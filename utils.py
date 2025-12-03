# utils.py
import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.lib import colors  

# ---------------- Feature lists (keep in training order) ----------------
FEATURES_DS2 = [
    "Age",
    "TT_Doses",
    "Gestational_Age",
    "Weight",
    "VDRL",
    "HBsAg",
    "Systolic_BP",
    "Diastolic_BP",
]


FEATURES_DS3 = [
    "Age",
    "Diastolic",
    "BS",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate",
]

def apply_global_css():
    st.markdown(
        """
        <style>
        /* ---- Global ---- */
        .main-title {
            font-size: 40px;
            font-weight: 800;
            color: #2E86C1;
            text-align: center;
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
        .welcome-text {
            font-size: 18px;
            font-weight: 400;
            text-align: center;
            color: #555555;
            margin-bottom: 1.5rem;
        }

        /* ---- Generic card ---- */
        .card {
            padding: 1.3rem 1.4rem;
            border-radius: 14px;
            border: 1px solid #dde3ec;
            background-color: #ffffffdd;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            margin-bottom: 1.2rem;
        }
        .card-header {
            font-size: 19px;
            font-weight: 650;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .card-header span.icon {
            font-size: 22px;
        }

        /* ---- Result card ---- */
        .result-card {
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            border: 1px solid #d4e6f1;
            background: linear-gradient(135deg, #ffffff 0%, #ebf5fb 45%, #fef9e7 100%);
            box-shadow: 0 6px 18px rgba(46, 134, 193, 0.15);
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }
        .result-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 0.6rem;
            color: #154360;
        }

        /* ---- Badge for risk ---- */
        .risk-badge {
            display: inline-block;
            padding: 0.32rem 0.9rem;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 600;
            color: white;
            letter-spacing: 0.02em;
        }

        .risk-low {
            background: linear-gradient(135deg, #1abc9c, #2ecc71);
        }
        .risk-moderate {
            background: linear-gradient(135deg, #f1c40f, #f39c12);
        }
        .risk-high {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }

        .risk-main-value {
            font-size: 28px;
            font-weight: 800;
            margin-top: 0.4rem;
        }
        .risk-subtext {
            font-size: 13px;
            color: #444;
        }

        /* ---- Section titles ---- */
        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.2rem;
        }
        .section-caption {
            font-size: 13px;
            color: #666;
            margin-bottom: 0.6rem;
        }

        /* Make SHAP plots sit on white cards */
        .shap-card {
            padding: 0.5rem 0.7rem 0.3rem 0.7rem;
            border-radius: 14px;
            background-color: #ffffffdd;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



# ---------------- Model loading ----------------
@st.cache_resource
def load_models():
    with open("best_xgbc_modelds2.pkl", "rb") as f:
        model_ds2 = pickle.load(f)
    with open("best_xgbc_model3.pkl", "rb") as f:
        model_ds3 = pickle.load(f)
    return model_ds2, model_ds3

# ---------------- SHAP helpers ----------------
def get_shap_values(model, x_array, predicted_class_index=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_array)

    if isinstance(shap_values, list):
        if predicted_class_index is None:
            predicted_class_index = 0
        shap_instance = shap_values[predicted_class_index][0]
        base_value = (
            explainer.expected_value[predicted_class_index]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_instance = shap_values[0]
        base_value = explainer.expected_value

    return shap_instance, base_value


def plot_shap_bar(shap_values, feature_names, title):
    idx_sorted = np.argsort(np.abs(shap_values))
    shap_sorted = shap_values[idx_sorted]
    feat_sorted = np.array(feature_names)[idx_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(feat_sorted, shap_sorted)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, base_value, x_row, feature_names, title):
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=x_row,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.plots.waterfall(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    return fig

# ---------------- PDF report ----------------
def create_pdf_report(model_name, input_dict, pred_label, proba_dict=None, shap_contribs=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # ---------- Title ----------
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Maternal Risk Prediction Report")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Model: {model_name}")
    y -= 15
    c.drawString(50, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "This report is a decision-support tool and does not replace clinical judgement.")
    y -= 25

    # ---------- Risk summary ----------
    lower_label = str(pred_label).lower()
    is_high = "high" in lower_label
    is_low = "low" in lower_label

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "1. Summary")
    y -= 20

    # Colored risk label
    c.setFont("Helvetica-Bold", 11)
    if is_high:
        c.setFillColor(colors.red)
    elif is_low:
        c.setFillColor(colors.green)
    else:
        c.setFillColor(colors.orange)

    c.drawString(60, y, f"Predicted risk level: {pred_label}")
    c.setFillColor(colors.black)
    y -= 20

    c.setFont("Helvetica", 10)
    if is_high:
        c.drawString(60, y, "The model suggests a HIGH level of maternal risk for this patient.")
        y -= 15
        c.setFillColor(colors.red)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, "⚠ URGENT:")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        y -= 15
        c.drawString(75, y, "The patient should consult a qualified doctor or health professional AS SOON AS POSSIBLE.")
        y -= 15
    elif is_low:
        c.drawString(60, y, "The model suggests a LOW level of maternal risk based on the provided inputs.")
        y -= 15
        c.drawString(60, y, "Routine monitoring and healthy lifestyle measures are still important.")
        y -= 15
    else:
        c.drawString(60, y, "The model suggests a MODERATE level of maternal risk.")
        y -= 15
        c.drawString(60, y, "Closer monitoring and review of modifiable risk factors are recommended.")
        y -= 15

    # ---------- Optional: class probabilities ----------
    if proba_dict:
        if y < 100:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "2. Class probabilities")
        y -= 20

        c.setFont("Helvetica", 10)
        for cls, p in proba_dict.items():
            c.drawString(60, y, f"{cls}: {p:.3f}")
            y -= 15
            if y < 80:
                c.showPage()
                y = height - 50

    # ---------- Input features ----------
    if y < 120:
        c.showPage()
        y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "3. Input information used by the model")
    y -= 20

    c.setFont("Helvetica", 10)
    for k, v in input_dict.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 15
        if y < 80:
            c.showPage()
            y = height - 50

    # ---------- Top SHAP contributions ----------
    if shap_contribs:
        if y < 120:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "4. Features that most influenced this prediction")
        y -= 20

        c.setFont("Helvetica", 10)
        c.drawString(
            60,
            y,
            "Positive values increased the estimated risk; negative values reduced it.",
        )
        y -= 20

        for feat, val in shap_contribs:
            c.drawString(60, y, f"{feat}: SHAP = {val:.4f}")
            y -= 15
            if y < 80:
                c.showPage()
                y = height - 50

    # ---------- Final note ----------
    if y < 100:
        c.showPage()
        y = height - 50

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "5. Important notice")
    y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(
        60,
        y,
        "This report is generated by a machine learning model and is intended for use by trained health professionals."
    )
    y -= 12
    c.drawString(
        60,
        y,
        "It should not be used as the sole basis for diagnosis or treatment decisions."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

    # utils.py (add imports if not there)

# --- Pretty label for risk ---
def format_risk_label(raw_label: str) -> str:
    s = str(raw_label).strip()
    lower = s.lower()
    # Common patterns
    if lower in ["0", "low"]:
        return "Low risk"
    if lower in ["1", "high"]:
        return "High risk"
    if lower in ["2", "medium", "moderate"]:
        return "Moderate risk"
    # Fallback: just capitalize
    return s.capitalize()

