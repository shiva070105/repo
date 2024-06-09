import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Function to generate sample data
def generate_data(num_samples):
    np.random.seed(0)
    X = np.random.randn(num_samples, 2)
    return X

# Function for K-means clustering
def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, y_kmeans

# Function for EM (Gaussian Mixture Model) clustering
def em_clustering(X, num_components):
    gmm = GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(X)
    y_gmm = gmm.predict(X)
    return gmm.means_, y_gmm

# Main Streamlit app
def main():
    st.title('Clustering Algorithms Demo')

    # Sidebar to input parameters
    st.sidebar.header('Parameters')
    num_samples = st.sidebar.slider('Number of samples', min_value=50, max_value=200, value=100)
    num_clusters = st.sidebar.slider('Number of clusters (K-means)', min_value=2, max_value=5, value=3)
    num_components = st.sidebar.slider('Number of components (Gaussian Mixture)', min_value=2, max_value=5, value=3)

    # Generate sample data
    X = generate_data(num_samples)

    # Perform K-means clustering
    kmeans_centers, kmeans_labels = kmeans_clustering(X, num_clusters)

    # Perform EM (Gaussian Mixture Model) clustering
    em_means, em_labels = em_clustering(X, num_components)

    # Display results
    st.subheader('K-means Clustering')
    st.write('Centroids:')
    st.write(kmeans_centers)
    st.write('Cluster labels:')
    st.write(kmeans_labels)

    st.subheader('EM (Gaussian Mixture Model) Clustering')
    st.write('Means:')
    st.write(em_means)
    st.write('Cluster labels:')
    st.write(em_labels)

if __name__ == '__main__':
    main()
