import streamlit as st
import numpy as np

# Function to generate synthetic data
def generate_data(num_samples, num_features):
    np.random.seed(0)
    X = np.random.randn(num_samples, num_features)
    return X

# Function for basic K-means clustering
def kmeans_clustering(X, num_clusters, max_iter=100):
    np.random.seed(0)
    centroids = X[np.random.choice(range(len(X)), num_clusters, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(num_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Main Streamlit app
def main():
    st.title('Clustering Algorithms Demo')

    # Sidebar to input parameters
    st.sidebar.header('Parameters')
    num_samples = st.sidebar.slider('Number of samples', min_value=50, max_value=200, value=100)
    num_features = st.sidebar.slider('Number of features', min_value=2, max_value=5, value=2)
    num_clusters = st.sidebar.slider('Number of clusters (K-means)', min_value=2, max_value=5, value=3)

    # Generate synthetic data
    X = generate_data(num_samples, num_features)

    # Perform KMeans clustering
    kmeans_centers, kmeans_labels = kmeans_clustering(X, num_clusters)

    # Display K-means results
    st.subheader('CLOUD STROMS - K-means Clustering')
    st.write('Centroids:')
    st.write(kmeans_centers)

    # Plotting the clusters using Streamlit's built-in chart
    import pandas as pd

    data = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
    data['Cluster'] = kmeans_labels

    if num_features >= 2:
        st.write('Scatter plot of the first two features:')
        st.write(st.altair_chart(
            alt.Chart(data).mark_circle(size=60).encode(
                x='Feature 1', y='Feature 2', color='Cluster:N'
            ).interactive()
        ))
    else:
        st.write('Histogram of the first feature:')
        st.write(st.altair_chart(
            alt.Chart(data).mark_bar().encode(
                x='Feature 1', y='count()', color='Cluster:N'
            ).interactive()
        ))

if __name__ == '__main__':
    main()
