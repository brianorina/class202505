import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Title
st.title("Breast Cancer Diagnosis Prediction App\nby AI Engineer Brian Nyakambi")

st.write("This app predicts whether a tumor is **benign** or **malignant** based on 30 input features.")

# Define the feature names (same order as training data)
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Create input fields dynamically
inputs = []
st.subheader("Enter the values for the 30 features:")
for feature in feature_names:
    value = st.number_input(f"{feature.title()}", value=0.0, format="%.5f")
    inputs.append(value)

# Convert input to numpy array
input_array = np.array(inputs).reshape(1, -1)

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("The model predicts: **Malignant Tumor**")
    else:
        st.success("The model predicts: **Benign Tumor**")
