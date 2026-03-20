import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE

@st.cache_resource
def load_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df.drop(columns=['customerID','gender','SeniorCitizen','Dependents','Partner'], inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    df = pd.get_dummies(df, columns=[
        'DeviceProtection','PhoneService','MultipleLines','OnlineSecurity',
        'OnlineBackup','TechSupport','StreamingTV','StreamingMovies',
        'PaperlessBilling','PaymentMethod'
    ], drop_first=True)

    encode = OrdinalEncoder(categories=[
        ['No','DSL','Fiber optic'],
        ['Month-to-month','One year','Two year']
    ])
    df[['InternetService','Contract']] = encode.fit_transform(df[['InternetService','Contract']])

    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_res)

    return model, scaler, X.columns

model, scaler, columns = load_model()
