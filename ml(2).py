# # Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset. Determine the number of clusters using the elbow method.
# We want to divide these customers into different groups based on their buying behaviour.
# This is called Clustering.
# Clustering is an Unsupervised Machine Learning technique because we don't provide predefined group names to the computer.

# Cell 1 – Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Cell 2 – Load dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")
print(df.head())
print(df.shape)

# Cell 3 – Preprocess the data
df = df.dropna()
df = df.drop_duplicates()
print(df.shape)

# Cell 4 – Select numerical columns
data = df[['QUANTITYORDERED', 'PRICEEACH', 'SALES']]
print(data.head())

# Cell 5 – Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Cell 6 – Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Cell 7 – Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X)
print(df[['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'KMeans_Cluster']].head())

# Cell 8 – Visualize K-Means clusters
plt.scatter(data['QUANTITYORDERED'],
            data['SALES'],
            c=df['KMeans_Cluster'])

plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.title("K-Means Clustering")
plt.show()

# Cell 9 – Hierarchical clustering
plt.figure(figsize=(10, 5))
Z = linkage(X[:100], method='ward')
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Cell 10 – Apply Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Hierarchical_Cluster'] = hc.fit_predict(X)
print(df[['QUANTITYORDERED', 'SALES', 'Hierarchical_Cluster']].head())

# Cell 11 – Display final result
print(df[['QUANTITYORDERED',
          'PRICEEACH',
          'SALES',
          'KMeans_Cluster',
          'Hierarchical_Cluster']].head(10))

# 
# Question	Easy answer
# What is clustering?	Clustering is grouping similar data points together.
# What is K-Means?	K-Means is an unsupervised learning algorithm that divides data into K clusters.
# What is Hierarchical clustering?	It groups data in a tree-like structure.
# What is the Elbow method?	It helps determine the optimal number of clusters.
# What is WCSS?	Within Cluster Sum of Squares measures the distance of points from their cluster centres.
# What is a centroid?	The centre point of a cluster.
# What is StandardScaler?	It converts features to a similar scale.
# What is a dendrogram?	A tree-like diagram showing how clusters are merged.
# What is unsupervised learning?	Learning from data without predefined output labels.
# Difference between K-Means and Hierarchical?	K-Means uses a fixed number of clusters, while Hierarchical builds a hierarchy of clusters.









