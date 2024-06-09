import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Function to generate synthetic data
def generate_data(num_samples, num_features):
    np.random.seed(0)
    X = np.random.randn(num_samples, num_features)
    return X

# Function to perform KMeans clustering
def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    return kmeans.cluster_centers_, y_kmeans

# Function to perform Gaussian Mixture Model clustering
def gmm_clustering(X, num_components):
    gmm = GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(X)
    y_gmm = gmm.predict(X)
    return gmm.means_, y_gmm

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
