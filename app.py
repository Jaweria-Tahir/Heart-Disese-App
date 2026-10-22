import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_files():
    model = pickle.load(open(BASE_DIR / "heart_svc_model.pkl", "rb"))
    encoders = pickle.load(open(BASE_DIR / "heart_encoders.pkl", "rb"))
    return model, encoders


model, encoders = load_files()

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below. Model: Support Vector Classifier trained on the "
         "UCI Cleveland Heart Disease dataset using 4 input attributes.")

# ---- Inputs (same 4 attributes and same text values as the notebook) ----
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox(
    "Chest pain type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
)
exercise_angina = st.selectbox("Exercise induced angina (chest pain during exercise)", ["No", "Yes"])
max_heart_rate = st.number_input("Maximum heart rate achieved (beats per minute)",
                                 min_value=60, max_value=220, value=150, step=1)

# ---- Predict ----
if st.button("Predict"):
    # Step 8.2: feature vector (same format as the notebook)
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MaxHeartRate": [max_heart_rate],
    })

    # Step 8.3: label encoding with the SAME encoders used in training
    features = user_input.copy()
    for column in ["Sex", "ChestPain", "ExerciseAngina"]:
        features[column] = encoders[column].transform(user_input[column])

    # Step 8.5: prediction
    predicted_class = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    label = encoders["HeartDisease"].inverse_transform([predicted_class])[0]
    confidence = round(float(max(probability)) * 100, 1)

    if label == "Present":
        st.error(f"Prediction: HEART DISEASE PRESENT  (confidence {confidence}%)")
    else:
        st.success(f"Prediction: HEART DISEASE ABSENT  (confidence {confidence}%)")
