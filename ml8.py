import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Define the Iris dataset with the first 10 data points
iris_data = {
    "data": np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
    ]),
    "target": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data["data"], iris_data["target"], random_state=0)

# Define a K-Nearest Neighbors classifier manually
class KNeighborsClassifierCustom:
    def _init_(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            prediction = np.argmax(np.bincount(self.y_train[nearest_neighbors]))
            predictions.append(prediction)
        return np.array(predictions)
    
# Initialize and train the KNeighborsClassifierCustom
kn = KNeighborsClassifierCustom(n_neighbors=1)
kn.fit(X_train, y_train)

# Streamlit app title
st.title("Iris Dataset KNN Classifier")

# Display the dataset
st.write("### Iris Dataset")
iris_df = pd.DataFrame(iris_data["data"], columns=["sepal length", "sepal width", "petal length", "petal width"])
iris_df['target'] = iris_data["target"]
st.write(iris_df.head())

# Display predictions
st.write("### Predictions")
results = []
for i in range(len(X_test)):
    x = X_test[i]
    prediction = kn.predict([x])
    result = {
        "Target": y_test[i],
        "Predicted": prediction[0],
    }
    results.append(result)
predictions_df = pd.DataFrame(results)
st.write(predictions_df)

# Calculate accuracy manually
correct = 0
for i in range(len(X_test)):
    x = X_test[i]
    prediction = kn.predict([x])[0]
    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
