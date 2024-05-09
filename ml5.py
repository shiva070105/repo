import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def preprocess_and_train(data):
    try:
        # Display the first 5 rows of the data
        st.write("First 5 rows of data:")
        st.write(data.head())

        # Obtain features (X) and target variable (y)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert categorical features into numerical values
        label_encoders = {}
        for column in X.columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])

        # Convert the target variable into numerical values
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Initialize and train the Gaussian Naive Bayes classifier
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Calculate the accuracy of the classifier on the test set
        accuracy = accuracy_score(classifier.predict(X_test), y_test)
        st.write("Accuracy:", accuracy)

    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.title("Tennis Data Classifier")

    # Allow user to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        preprocess_and_train(data)

if __name__ == "__main__":
    main()
