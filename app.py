import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------
# 1. Load dataset
# ----------------------
df = pd.read_csv("products_realistic.csv")  # Make sure CSV is in your repo

X = df[['Brand', 'RAM', 'Storage', 'Processor']]
y = df['Price']

# ----------------------
# 2. Build pipeline
# ----------------------
categorical_features = ['Brand', 'RAM', 'Storage', 'Processor']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X, y)

# ----------------------
# 3. Streamlit UI
# ----------------------
st.title("Laptop Price Prediction Model")

brand = st.selectbox("Brand", df['Brand'].unique())
ram = st.selectbox("RAM", df['RAM'].unique())
storage = st.selectbox("Storage", df['Storage'].unique())
processor = st.selectbox("Processor", df['Processor'].unique())

if st.button("Predict Price"):
    input_data = pd.DataFrame([[brand, ram, storage, processor]],
                              columns=['Brand', 'RAM', 'Storage', 'Processor'])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹{predicted_price:,.2f}")
