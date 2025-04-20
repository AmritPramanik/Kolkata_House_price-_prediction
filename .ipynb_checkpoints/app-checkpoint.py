import streamlit as st
import pandas as pd
import joblib
from flask import Flask, render_template

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
# Load preprocessor and model
column_trans = joblib.load(open('HouseData.pkl','rb'))
model = joblib.load(open('LinearRegressionModel.pkl', 'rb'))

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction App")

st.markdown("""
    <style>
        /* Remove top padding and adjust spacing */
        .block-container {
            padding-top: 50px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("Enter the house details below to predict its estimated price (in ₹ )")

# --- User Input ---
# Ideally, populate from actual data or metadata
house_data = pd.read_pickle("HouseData.pkl")

# Extract unique values
locations = sorted(house_data['Location'].dropna().unique())
house_types = sorted(house_data['House_type'].dropna().unique())

print(locations)
print(house_types)


house_types = ['Apartment', 'House']

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", locations)
with col2:
    house_type = st.selectbox("House/Apartment", house_types)

# Second row: BHK and Area
col3, col4 = st.columns(2)
with col3:
    bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)
with col4:
    area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, step=50)


# --- Predict Button ---
if st.button("Predict Price"):
    # You may need to encode 'location' and 'house_type' same as during training
    input_df = pd.DataFrame([{
        'Location': location,
        'House_type': house_type,
        'BHK': bhk,
        'Area(sqft)': area_sqft
    }])

    # If your model needs preprocessing (e.g., encoding), add here

    prediction = model.predict(input_df)*10000000
    st.success(f"Estimated Price: ₹ {prediction[0]:,.0f}")

