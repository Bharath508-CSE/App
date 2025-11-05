# app.py
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------------
# 🎯 Load or Create Sample Dataset
# -------------------------------------
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'bedrooms': np.random.randint(1, 6, 500),
        'bathrooms': np.random.uniform(1, 4, 500).round(1),
        'sqft_living': np.random.randint(500, 5000, 500),
        'sqft_lot': np.random.randint(1000, 20000, 500),
        'floors': np.random.randint(1, 3, 500),
        'grade': np.random.randint(1, 13, 500),
        'zipcode': np.random.randint(100000, 999999, 500),
        # Generate price roughly in INR (lakhs)
        'price': np.random.randint(20_00_000, 3_00_00_000, 500)
    })
    return data

data = load_data()

# -------------------------------------
# ⚙️ Train XGBoost Model
# -------------------------------------
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'zipcode']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    n_estimators=120,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# -------------------------------------
# 🎨 Streamlit UI
# -------------------------------------
st.set_page_config(page_title="🏡 House Price Prediction (INR)", layout="centered")

st.title("🏠 House Price Prediction")
st.markdown("### Give features of the house you wish")

# Input features
bedrooms = st.number_input("No. of bedrooms in the house", 1, 10, 3)
bathrooms = st.number_input("No. of bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.slider("Select the living area (in sqft):", 500, 10000, 1500)
sqft_lot = st.slider("Select the lot size (in sqft):", 500, 50000, 7500)
floors = st.number_input("Enter number of floors", 1, 3, 1)
grade = st.slider("Select the house grade (1-13)", 1, 13, 7)
zipcode = st.number_input("Enter the zipcode (numeric only)", 100000, 999999, 560001)

# Prediction button
if st.button("💸 Predict Price"):
    st.info("Running prediction...")
    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, grade, zipcode]])
    predicted_price = xgb_model.predict(features)[0]

    # Format price nicely in INR
    price_in_lakhs = predicted_price / 1_00_000
    st.success(f"🏡 **Estimated House Price:** ₹{predicted_price:,.0f}  \n💰 _(~{price_in_lakhs:.2f} lakhs)_")
