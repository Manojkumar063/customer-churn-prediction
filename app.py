import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle

# Load the trained model
with open("xgboost_churn_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical columns
    categorical_columns = ["ContractType", "TechSupport", "InternetService", "PaperlessBilling", "PaymentMethod"]
    input_df = pd.get_dummies(input_df, columns=categorical_columns)

    # Ensure column order matches the training data
    model_features = loaded_model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    return input_df

def predict_churn(input_data):
    input_df = preprocess_input(input_data)
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0][1]
    return prediction, probability

# Streamlit UI
st.title("Customer Churn Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
average_monthly_charges = st.number_input("Average Monthly Charges", min_value=0.0, value=50.0)
customer_lifetime_value = st.number_input("Customer Lifetime Value", min_value=0.0, value=5000.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Predict button
if st.button("Predict Churn"):
    input_data = {
        "Age": age,
        "Gender": gender,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "TechSupport": tech_support,
        "Tenure": tenure,
        "PaperlessBilling": paperless_billing,
        "AverageMonthlyCharges": average_monthly_charges,
        "CustomerLifetimeValue": customer_lifetime_value,
        "ContractType": contract_type,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    # Get prediction
    churn_prediction, churn_prob = predict_churn(input_data)

    # Display results
    st.subheader("Prediction Result")
    if churn_prediction == 1:
        st.error(f"The customer is **likely to churn** (Probability: {churn_prob:.2f})")
    else:
        st.success(f"The customer is **not likely to churn** (Probability: {churn_prob:.2f})")
