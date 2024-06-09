from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd

# Verify installation of scikit-learn
try:
  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
except ImportError as e:
  st.error("Error importing scikit-learn packages. Please ensure scikit-learn is installed in your environment.")
  raise e

# Function to encode features
def encode_features(df, columns):
  encoders = {}
  for column in columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder
  return df, encoders

# Load data from CSV
data = pd.read_csv('tennisdata.csv')
st.write("The first 5 rows of the dataset are:")
st.write(data.head())

# Separate features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

st.write("The first 5 rows of the features are:")
st.write(X.head())
st.write("The first 5 values of the target are:")
st.write(y.head())

# Encode the categorical features and target
X, feature_encoders = encode_features(X, X.columns)

# Encode the target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

st.write("Encoded features:")
st.write(X.head())
st.write("Encoded target:")
st.write(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train the Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy of the model:", accuracy)
