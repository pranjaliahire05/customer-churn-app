import streamlit as st
import pickle
import pandas as pd

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

# Inputs
InternetService = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tenure = st.number_input("Tenure", 0, 100)
MonthlyCharges = st.number_input("Monthly Charges", 0.0)
TotalCharges = st.number_input("Total Charges", 0.0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

# Prepare input
input_data = {col: 0 for col in columns}

input_data['InternetService'] = ["No","DSL","Fiber optic"].index(InternetService)
input_data['Contract'] = ["Month-to-month","One year","Two year"].index(Contract)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = MonthlyCharges
input_data['TotalCharges'] = TotalCharges

if PhoneService == "Yes":
    input_data['PhoneService_Yes'] = 1

if PaperlessBilling == "Yes":
    input_data['PaperlessBilling_Yes'] = 1

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"❌ Customer will churn ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Customer will not churn ({(1-prob)*100:.2f}%)")