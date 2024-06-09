import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Define the Iris dataset with the first 15 data points
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
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.8, 4.0, 1.2, 0.2],
    ]),
    "target": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris_data["data"], iris_data["target"], test_size=0.3, random_state=0)

# Initialize and train the KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Streamlit app title
st.title("Iris Dataset KNN Classifier")

# Display the dataset
st.write("### Iris Dataset")
iris_df = pd.DataFrame(iris_data["data"], columns=["sepal length", "sepal width", "petal length", "petal width"])
iris_df['target'] = iris_data["target"]
st.write(iris_df)

# Display predictions
st.write("### Predictions")
results = []
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    result = {
        "Target": y_test[i],
        "Predicted": prediction[0],
    }
    results.append(result)
predictions_df = pd.DataFrame(results)
st.write(predictions_df)

# Display accuracy
accuracy = kn.score(X_test, y_test)
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
