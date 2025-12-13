import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Page configuration
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.markdown(""This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
Enter customer details below to get a prediction.
"")

# Load model
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id='SharleyK/tourism-package-model',
            filename='best_model.pkl'
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Create input form
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
    occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])

with col2:
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Followups", min_value=0.0, value=3.0)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])

with col3:
    num_trips = st.number_input("Number of Trips Per Year", min_value=0.0, value=2.0)
    passport = st.selectbox("Has Passport", ['Yes', 'No'])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns Car", ['Yes', 'No'])
    num_children = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)

col4, col5 = st.columns(2)
with col4:
    designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
with col5:
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])

# Prediction button
if st.button("Predict Purchase Probability", type="primary"):
    if model is not None:
        # Create input dataframe
        # Note: Adjust feature order and encoding based on your actual model
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [1 if type_of_contact == 'Company Invited' else 0],
            'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [['Salaried', 'Freelancer', 'Small Business', 'Large Business'].index(occupation)],
            'Gender': [0 if gender == 'Male' else 1],
            'NumberOfPersonVisiting': [num_persons],
            'NumberOfFollowups': [num_followups],
            'ProductPitched': [['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'].index(product_pitched)],
            'PreferredPropertyStar': [preferred_star],
            'MaritalStatus': [['Single', 'Married', 'Divorced', 'Unmarried'].index(marital_status)],
            'NumberOfTrips': [num_trips],
            'Passport': [1 if passport == 'Yes' else 0],
            'PitchSatisfactionScore': [pitch_satisfaction],
            'OwnCar': [1 if own_car == 'Yes' else 0],
            'NumberOfChildrenVisiting': [num_children],
            'Designation': [['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'].index(designation)],
            'MonthlyIncome': [monthly_income]
        })

        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.success("Prediction Complete!")

            col_pred1, col_pred2 = st.columns(2)

            with col_pred1:
                st.metric("Prediction", "Will Purchase" if prediction == 1 else "Will Not Purchase")

            with col_pred2:
                st.metric("Confidence", f"{max(probability)*100:.2f}%")

            # Recommendation
            if prediction == 1:
                st.balloons()
                st.info("🎯 **Recommendation:** This customer has a high likelihood of purchasing. Consider prioritizing this lead!")
            else:
                st.warning("💡 **Recommendation:** This customer may need more engagement. Consider additional followups or tailored offers.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model not loaded. Please check the configuration.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Powered by Visit with Us")
