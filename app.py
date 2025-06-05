import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os
print("Current working directory:", os.getcwd())

# Load trained model and scaler
model = joblib.load("pipe.pkl")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Restaurant Rating Predictor", layout="centered")
st.title("🍽️ Restaurant Rating Predictor")
st.markdown("Enter restaurant details to predict the rating")

# UI for input fields
average_cost = st.number_input("Average Cost (₹)", min_value=0.0, step=1.0)
minimum_order = st.number_input("Minimum Order (₹)", min_value=0.0, step=1.0)
votes = st.number_input("Votes", min_value=0, step=1)
reviews = st.number_input("Reviews", min_value=0, step=1)
delivery_time = st.number_input("Delivery Time (in minutes)", min_value=0, step=1)
location_encoded = st.number_input("Location Encoded (avg rating for location)", min_value=0.0, step=0.1)
cuisines_encoded = st.number_input("Cuisines Encoded (avg rating for cuisines)", min_value=0.0, step=0.1)
cuisine_count = st.slider("Number of Cuisines", 1, 10, 2)

pipe = pickle.load(open('pipe.pkl', "rb"))
# Predict button
if st.button("Predict Rating"):
    if minimum_order == 0:
        st.error("Minimum Order must be greater than 0 to avoid division by zero")
    else:
        cost_per_order = average_cost / minimum_order
        log_votes = np.log1p(votes)
        log_reviews = np.log1p(reviews)
        engagement = log_votes * log_reviews

        input_features = pd.DataFrame({
            'Average_Cost': [average_cost],
            'Minimum_Order': [minimum_order],
            'Votes': [log_votes],
            'Reviews': [log_reviews],
            'Delivery_Time': [delivery_time],
            'Cost_Per_Order': [cost_per_order],
            'Engagement': [engagement],
            'Num_Cuisines': [cuisine_count],
            'Location_Encoded': [location_encoded],
            'Cuisines_Encoded': [cuisines_encoded]
        })

        input_scaled = scaler.transform(input_features)
        rating_pred = model.predict(input_scaled)[0]

        st.success(f"⭐ Predicted Rating: {rating_pred:.2f} / 5.0")