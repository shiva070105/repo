import streamlit as st
import pandas as pd
import numpy as np
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

    # Manually split the data into training and testing sets
    np.random.seed(42)
    mask = np.random.rand(len(heart_disease)) < 0.8
    train_data = heart_disease[mask]
    test_data = heart_disease[~mask]

    X_train = train_data.drop('heartdisease', axis=1)
    y_train = train_data['heartdisease']
    X_test = test_data.drop('heartdisease', axis=1)
    y_test = test_data['heartdisease']

    # Train a logistic regression model (you can implement your own if needed)
    def logistic_regression(X_train, y_train):
        # Implement logistic regression from scratch (example)
        # Initialize weights and biases
        weights = np.zeros(X_train.shape[1])
        bias = 0
        lr = 0.01  # Learning rate
        epochs = 1000

        # Gradient descent
        for epoch in range(epochs):
            # Compute predictions
            logits = np.dot(X_train, weights) + bias
            y_pred = 1 / (1 + np.exp(-logits))

            # Compute gradients
            dw = (1 / X_train.shape[0]) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / X_train.shape[0]) * np.sum(y_pred - y_train)

            # Update parameters
            weights -= lr * dw
            bias -= lr * db

        return weights, bias

    # Train the logistic regression model
    weights, bias = logistic_regression(X_train.values, y_train.values)

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
    def predict_logistic_regression(weights, bias, user_input):
        logits = np.dot(user_input.values, weights) + bias
        y_pred = 1 / (1 + np.exp(-logits))
        return y_pred

    if st.button("Predict"):
        prediction = predict_logistic_regression(weights, bias, user_input)
        st.write("Prediction for Heart Disease:")
        st.write('Yes' if prediction[0] >= 0.5 else 'No')
else:
    st.write("Please upload a CSV file to proceed.")
