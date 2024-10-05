import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')  # Load your encoder if applicable

# Function to preprocess the input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables if needed
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    existing_categorical_cols = [col for col in categorical_cols if col in input_df.columns]

    if existing_categorical_cols:
        encoded_df = pd.DataFrame(encoder.transform(input_df[existing_categorical_cols]), 
                                  columns=encoder.get_feature_names_out(existing_categorical_cols))
        
        input_df = input_df.drop(existing_categorical_cols, axis=1)
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return input_df

# Streamlit app layout
st.title("Car Price Prediction App")

# Input fields for the user
year = st.number_input("Year", min_value=1900, max_value=2024, value=2021)
km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
fuel = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", options=['Dealer', 'Individual'])
transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

# Button to predict the price
if st.button("Predict Price"):
    input_data = {
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }
    
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    
    st.success(f"The predicted price of the car is: ₹{prediction[0]:,.2f}")

