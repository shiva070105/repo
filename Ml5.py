import streamlit as st
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Display information about the dataset
st.write("Description of the Iris dataset:")
st.write(iris.DESCR)

# ... (rest of your Streamlit app code using the iris data)


# Sidebar
st.sidebar.title('Naive Bayes Classifier')
st.sidebar.markdown('Select the classifier and parameters.')

# Display dataset
if st.sidebar.checkbox('Show raw data'):
    st.subheader('Iris Dataset (First 10 rows)')
    st.write(X.head(10))
    st.write(f'Target names: {iris.target_names}')

# Model Selection
classifier_name = st.sidebar.selectbox('Select Classifier', ['Gaussian Naive Bayes'])
test_size = st.sidebar.slider('Test set size (%)', 10, 50, 30, 5)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=0)

# Model training
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Model evaluation
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Display classification report
st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display predictions
if st.checkbox('Show predictions'):
    st.subheader('Predictions')
    prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    prediction_df['Correct'] = prediction_df['Actual'] == prediction_df['Predicted']
    st.write(prediction_df)

# Sidebar Footer
st.sidebar.markdown("""
[Get the source code](https://github.com/streamlit/demo-iris)
""")

# Footer
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
