import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Generate some sample data
np.random.seed(0)
X = np.random.randn(100, 2)

# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Initialize GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Main Streamlit app
st.title('Clustering Algorithms Demo')

# Display K-means results
st.subheader('K-means Clustering')
st.write('Centroids:')
st.write(kmeans.cluster_centers_)
st.write('Cluster labels:')
st.write(y_kmeans)

# Display EM (Gaussian Mixture Model) results
st.subheader('EM (Gaussian Mixture Model) Clustering')
st.write('Means:')
st.write(gmm.means_)
st.write('Cluster labels:')
st.write(y_gmm)
