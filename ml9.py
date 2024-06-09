import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Title of the Streamlit app
st.title("Iris Dataset K-Nearest Neighbors Classifier")

# Load the Iris dataset
dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Train the K-Nearest Neighbors classifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Main Streamlit app
def main():
    # Display the test samples and their predicted and actual labels
    st.write("## Test Sample Predictions")
    for i in range(len(X_test)):
        x = X_test[i]
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        st.write(f"Sample {i+1}:")
        st.write(f"  - Target: {y_test[i]} ({dataset['target_names'][y_test[i]]})")
        st.write(f"  - Predicted: {prediction[0]} ({dataset['target_names'][prediction[0]]})")

    # Display the model accuracy
    accuracy = kn.score(X_test, y_test)
    st.write("## Model Accuracy")
    st.write(f"The accuracy of the K-Nearest Neighbors classifier is: {accuracy:.2f}")

if __name__ == '__main__':
    main()
