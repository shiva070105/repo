import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Define the Iris dataset manually
iris_data = {
    "data": np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        # ... (rest of the Iris data points)
        [6.8, 3.0, 5.0, 1.7],
    ]),
    "target": np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
}

# Convert data and target to pandas DataFrame
data = pd.DataFrame(iris_data["data"], columns=["Sepal length", "Sepal width", "Petal length", "Petal width"])
target = iris_data["target"]

# Split data into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)

# Define the KNeighborsClassifier model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn.fit(X_train, y_train)

# Title and description for the app
st.title("KNeighborsClassifier App")
st.write("This app allows you to explore K-Nearest Neighbors classification on the Iris dataset.")

# User input for prediction
user_input = st.text_input("Enter new data point (comma separated values):")

if user_input:
  # Convert user input to a list of floats
  user_data = np.array([float(x) for x in user_input.split(",")])

  # Reshape the user data for prediction
  user_data = user_data.reshape(1, -1)

  # Make prediction using the trained model
  prediction = knn.predict(user_data)[0]

  # Display prediction result
  st.write(f"Predicted class: {prediction}")

# Display model accuracy on test data
accuracy = knn.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")
