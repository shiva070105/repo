import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Streamlit app title
st.title("Heart Disease Prediction using Logistic Regression")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data from CSV
    heart_disease = pd.read_csv(uploaded_file)
    st.write("The first 5 values of the dataset:")
    st.write(heart_disease.head())

    # Preprocess the data
    label_encoders = {}
    categorical_columns = ['age', 'Gender', 'Family', 'diet', 'Lifestyle', 'cholestrol']
    for column in categorical_columns:
        le = LabelEncoder()
        heart_disease[column] = le.fit_transform(heart_disease[column])
        label_encoders[column] = le

    # Split the data into training and testing sets
    X = heart_disease.drop('heartdisease', axis=1)
    y = heart_disease['heartdisease']



    # User inputs
    st.write('### Enter the following details:')
    
    age = st.selectbox('Age', ['SuperSeniorCitizen', 'SeniorCitizen', 'MiddleAged', 'Youth', 'Teen'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    diet = st.selectbox('Diet', ['High', 'Medium'])
    lifestyle = st.selectbox('Lifestyle', ['Athlete', 'Active', 'Moderate', 'Sedentary'])
    cholestrol = st.selectbox('Cholesterol', ['High', 'BorderLine', 'Normal'])

    # Convert user inputs to appropriate format
    user_input = pd.DataFrame({
        'age': [age],
        'Gender': [gender],
        'Family': [family_history],
        'diet': [diet],
        'Lifestyle': [lifestyle],
        'cholestrol': [cholestrol]
    })
    
    for column in user_input.columns:
        user_input[column] = label_encoders[column].transform(user_input[column])

   

      
else:
    st.write("Please upload a CSV file to proceed.")
