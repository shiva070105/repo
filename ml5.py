import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    st.title("Tennis Data Classifier")

    # Allow user to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the first few rows of the uploaded data
        st.write("First few rows of the uploaded data:")
        st.write(data.head())

        # Obtain train data and train output
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Convert categorical features into numerical values
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        # Convert the target variable into numerical values
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Initialize and train the Gaussian Naive Bayes classifier
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Calculate the accuracy of the classifier on the test set
        accuracy = accuracy_score(classifier.predict(X_test), y_test)
        st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
