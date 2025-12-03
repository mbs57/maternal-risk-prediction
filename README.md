# 🩺 Maternal Risk Prediction App

A Streamlit-based machine learning web application for predicting **maternal health risk** and **pregnancy (antenatal) risk** using clinically relevant features.  
This tool helps health professionals assess patient status and understand *why* the model made a particular prediction through clear SHAP explanations.

---

## 🌟 Features

### 🔮 **Two Machine Learning Models**
1. **General Maternal Model**
   - Uses vitals and clinical history (BMI, Blood Sugar, complications, diabetes, etc.)
   - Outputs: Low / Moderate / High Risk
   - SHAP-based interpretability

2. **Pregnancy / Antenatal Model**
   - Uses gestational age, TT doses, infection markers, BP, and vitals
   - Outputs: Low / Moderate / High Risk
   - SHAP-based interpretability

---

### 🧠 **Explainability (XAI)**
Each prediction includes:

- Feature impact **bar plot**
- Waterfall plot showing step-by-step risk shift
- Top 3 contributing features summarized in text
- Clear “in simple terms” explanation

---

### 📄 **Downloadable PDF Report**
The app generates a professional PDF including:

- Risk level (color-coded)
- Clear summary
- Class probability table
- Input features used by the model
- Top SHAP feature contributions
- ⚠ For **high-risk predictions**, the PDF includes a **strong warning** advising the patient to consult a doctor immediately.

---

## 🚀 Deploy on Streamlit Cloud

If you want to deploy this project yourself:

1. Push the repo to GitHub  
2. Go to https://share.streamlit.io  
3. Click **New App**
4. Select:
   - Repository: `mbs57/maternal-risk-prediction`
   - Branch: `main`
   - Main file: `app.py`
5. Deploy!

Streamlit Cloud will automatically:
- install dependencies from `requirements.txt`
- run the app
- host it publicly

---

## 📁 Project Structure
├── app.py
├── utils.py
├── home_page.py
├── general_model_page.py
├── pregnancy_model_page.py
│
├── best_xgbc_model2.pkl # Pregnancy model
├── best_xgbc_model3.pkl # General maternal model
│
├── requirements.txt
└── README.md


---

## 🛠 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

### Main libraries:

streamlit

numpy

xgboost

shap

matplotlib

reportlab

scikit-learn

📊 Models

The two models (best_xgbc_model2.pkl and best_xgbc_model3.pkl) are trained offline and loaded automatically when the app runs.

Pregnancy Model → 8 features

General Maternal Model → multiple vitals + clinical history

👤 Author

Mrinal Basak Shuvo
Student | Developer | ML Enthusiast
GitHub: https://github.com/mbs57

⚠ Disclaimer

This tool is for decision support only.
It is not a replacement for medical diagnosis.
High-risk results should be followed by immediate consultation with a qualified healthcare professional. 

