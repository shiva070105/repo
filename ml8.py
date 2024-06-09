import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
        distances = cdist(X, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(num_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Function for basic Gaussian Mixture Model clustering (using K-means initialization)
def gmm_clustering(X, num_components, max_iter=100):
    centroids, labels = kmeans_clustering(X, num_components)
    for _ in range(max_iter):
        probabilities = np.array([np.exp(-cdist(X, [mean], 'euclidean')**2) for mean in centroids])
        probabilities = probabilities / probabilities.sum(axis=0)
        labels = np.argmax(probabilities, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(num_components)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Function to plot clustering results
def plot_clusters(X, labels, centers, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title('Clustering Algorithms Demo')

    # Sidebar to input parameters
    st.sidebar.header('Parameters')
    num_samples = st.sidebar.slider('Number of samples', min_value=50, max_value=200, value=100)
    num_features = st.sidebar.slider('Number of features', min_value=2, max_value=5, value=2)
    num_clusters = st.sidebar.slider('Number of clusters (K-means)', min_value=2, max_value=5, value=3)
    num_components = st.sidebar.slider('Number of components (Gaussian Mixture)', min_value=2, max_value=5, value=3)

    # Generate synthetic data
    X = generate_data(num_samples, num_features)

    # Perform KMeans clustering
    kmeans_centers, kmeans_labels = kmeans_clustering(X, num_clusters)

    # Perform Gaussian Mixture Model clustering
    gmm_means, gmm_labels = gmm_clustering(X, num_components)

    # Display K-means results
    st.subheader('K-means Clustering')
    st.write('Centroids:')
    st.write(kmeans_centers)
    plot_clusters(X, kmeans_labels, kmeans_centers, 'K-means Clustering', 'Feature 1', 'Feature 2')

    # Display EM (Gaussian Mixture Model) results
    st.subheader('EM (Gaussian Mixture Model) Clustering')
    st.write('Means:')
    st.write(gmm_means)
    plot_clusters(X, gmm_labels, gmm_means, 'EM (Gaussian Mixture Model) Clustering', 'Feature 1', 'Feature 2')

if __name__ == '__main__':
    main()
