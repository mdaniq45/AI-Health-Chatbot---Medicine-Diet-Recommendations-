import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load the dataset
df = pd.read_csv(r"C:\Users\hi\Downloads\Disease_symptom_and_patient_profile_dataset.csv")

# Convert "Yes"/"No" to 1/0
yes_no_columns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Encode categorical variables using separate LabelEncoders
gender_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
outcome_encoder = LabelEncoder()
disease_encoder = LabelEncoder()  # Separate encoder for Disease

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Blood Pressure"] = bp_encoder.fit_transform(df["Blood Pressure"])
df["Cholesterol Level"] = cholesterol_encoder.fit_transform(df["Cholesterol Level"])
df["Outcome Variable"] = outcome_encoder.fit_transform(df["Outcome Variable"])
df["Disease"] = disease_encoder.fit_transform(df["Disease"])  # Encode disease names

# Define features and target
X = df.drop(columns=["Disease", "Outcome Variable"])
y = df["Disease"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
pickle.dump(model, open("health_chatbot_model.pkl", "wb"))
pickle.dump(disease_encoder, open("disease_encoder.pkl", "wb"))  # Save only the disease encoder

print("✅ Model training completed successfully!")

# ----------------- Streamlit UI ----------------- #

# Load trained model & encoder
model = pickle.load(open("health_chatbot_model.pkl", "rb"))
disease_encoder = pickle.load(open("disease_encoder.pkl", "rb"))

# Streamlit UI
st.title("🩺 AI Health Chatbot - Medicine & Diet Recommendations")

st.sidebar.header("Enter Your Symptoms")
fever = st.sidebar.radio("Do you have fever?", ["Yes", "No"])
cough = st.sidebar.radio("Do you have cough?", ["Yes", "No"])
fatigue = st.sidebar.radio("Do you feel fatigued?", ["Yes", "No"])
breathing = st.sidebar.radio("Do you have difficulty breathing?", ["Yes", "No"])
age = st.sidebar.slider("Enter your age", 1, 100, 25)
gender = st.sidebar.radio("Select Gender", ["Male", "Female"])
bp = st.sidebar.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
cholesterol = st.sidebar.selectbox("Cholesterol Level", ["Low", "Normal", "High"])

# Convert categorical inputs to numerical
fever = 1 if fever == "Yes" else 0
cough = 1 if cough == "Yes" else 0
fatigue = 1 if fatigue == "Yes" else 0
breathing = 1 if breathing == "Yes" else 0
gender = gender_encoder.transform([gender])[0]  # Use saved encoder
bp = bp_encoder.transform([bp])[0]  # Use saved encoder
cholesterol = cholesterol_encoder.transform([cholesterol])[0]  # Use saved encoder

# Prediction
if st.sidebar.button("Get Health Recommendations"):
    input_data = np.array([[fever, cough, fatigue, breathing, age, gender, bp, cholesterol]])
    disease_pred = model.predict(input_data)[0]

    # Ensure predicted disease exists in the encoder
    if disease_pred in range(len(disease_encoder.classes_)):
        disease_name = disease_encoder.inverse_transform([disease_pred])[0]
    else:
        disease_name = "Unknown Disease"
        st.error("⚠️ Error: Unrecognized disease prediction. Please retrain the model with more data.")

    st.subheader(f"🦠 Predicted Disease: **{disease_name}**")

    # Medicine Recommendations
    medicine_recommendations = {
        "Common Cold": "Paracetamol, Antihistamines",
        "Flu": "Oseltamivir, Ibuprofen",
        "Asthma": "Inhalers, Bronchodilators",
        "Hypertension": "ACE inhibitors, Beta-blockers",
    }

    diet_recommendations = {
        "Common Cold": "Vitamin C-rich foods, Warm fluids",
        "Flu": "Herbal teas, Soup, Hydration",
        "Asthma": "Anti-inflammatory foods, Omega-3 fatty acids",
        "Hypertension": "Low-sodium diet, Leafy greens",
    }

    precautions = {
        "Common Cold": "Stay hydrated, Rest well",
        "Flu": "Wear masks, Avoid close contact",
        "Asthma": "Avoid allergens, Use inhalers regularly",
        "Hypertension": "Monitor BP regularly, Exercise daily",
    }

    st.write(f"💊 **Recommended Medicines:** {medicine_recommendations.get(disease_name, 'Consult a doctor')}")
    st.write(f"🥗 **Diet Suggestions:** {diet_recommendations.get(disease_name, 'Maintain a balanced diet')}")
    st.write(f"⚠️ **Precautions:** {precautions.get(disease_name, 'Consult a doctor for advice')}")

    st.success("✅ Stay healthy and take precautions!")
