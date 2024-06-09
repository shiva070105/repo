import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    # Define standard categorical values
    categorical_values = {
        'age': ['SuperSeniorCitizen', 'SeniorCitizen', 'MiddleAged', 'Youth', 'Teen'],
        'Gender': ['Male', 'Female'],
        'Family': ['Yes', 'No'],
        'diet': ['High', 'Medium'],
        'Lifestyle': ['Athlete', 'Active', 'Moderate', 'Sedentary'],
        'cholestrol': ['High', 'BorderLine', 'Normal']
    }
    
    for column in categorical_columns:
        le = LabelEncoder()
        le.fit(categorical_values[column])
        heart_disease[column] = le.transform(heart_disease[column])
        label_encoders[column] = le

    # Split the data into training and testing sets
    X = heart_disease.drop('heartdisease', axis=1)
    y = heart_disease['heartdisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # User inputs
    st.write('### Enter the following details:')
    
    age = st.selectbox('Age', categorical_values['age'])
    gender = st.selectbox('Gender', categorical_values['Gender'])
    family_history = st.selectbox('Family History', categorical_values['Family'])
    diet = st.selectbox('Diet', categorical_values['diet'])
    lifestyle = st.selectbox('Lifestyle', categorical_values['Lifestyle'])
    cholestrol = st.selectbox('Cholestrol', categorical_values['cholestrol'])

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

    # Predict heart disease
    if st.button("Predict"):
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        # Display the prediction
        st.write("Prediction for Heart Disease:")
        st.write('Yes' if prediction[0] == 1 else 'No')
        st.write("Prediction Probability:")
        st.write(f'No: {prediction_proba[0][0]:.2f}, Yes: {prediction_proba[0][1]:.2f}')
else:
    st.write("Please upload a CSV file to proceed.")
