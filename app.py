import streamlit as st
import pandas as pd
import joblib

# Load trained model
model_data = joblib.load("model.pkl")
pipeline = model_data["model"]
feature_names = model_data["features"]

st.title("💻 Laptop Price Prediction")

# Input form
brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus"])
ram = st.selectbox("RAM", ["8GB", "16GB", "32GB"])
storage = st.selectbox("Storage", ["256GB SSD", "512GB SSD", "1TB SSD", "1TB HDD"])
processor = st.selectbox("Processor", ["i3", "i5", "i7", "i9", "Ryzen 5", "Ryzen 7", "M1"])

if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame([[brand, ram, storage, processor]], columns=feature_names)

    # Prediction
    predicted_price = pipeline.predict(input_data)[0]
    st.success(f"Estimated Price: ₹{predicted_price:,.0f}")
