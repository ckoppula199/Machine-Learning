"""
K-Means Clustering model to determine patterns in customer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

"""
---------------Data Pre-processing---------------
"""
# Unsupervised algorithm means no need to split into a train and test set
# as there is no dependent variable
dataset = pd.read_csv('../Data/Clustering/Mall_Customers.csv')
# Removing customer_id column as it does not provide useful information
# Only using last 2 features so that results can be eaasily visualised
X = dataset.iloc[:, 3:].values

"""
---------------Using 'elbow method' to find the optimal number of clusters---------------
"""
# Within Cluster Sum of Squares used as metric to find optimal number of clusters
wcss = []
# Testing with upto 10 clusters
for num_clusters in range(1, 11):
    # k-means++ avoids random initialisation trap
    kmeans = KMeans(n_clusters = num_clusters, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the graph of wcss against cluster number
plt.figure(figsize=(16,8))
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum of Squares")
plt.show()


"""
---------------Training K-Means model on dataset---------------
"""
# Graph shows that 5 clusters is the optimal amount
kmeans = KMeans(n_clusters = 5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)


"""
---------------Visualising the Cluster Predictions---------------
"""
plt.figure(figsize=(16,8))
# Select rows such that only points plotted are the ones assigned to the same cluster
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], s=100, color='red', label='Cluster 1')
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], s=100, color='blue', label='Cluster 2')
plt.scatter(X[y_pred==2,0], X[y_pred==2,1], s=100, color='green', label='Cluster 3')
plt.scatter(X[y_pred==3,0], X[y_pred==3,1], s=100, color='cyan', label='Cluster 4')
plt.scatter(X[y_pred==4,0], X[y_pred==4,1], s=100, color='magenta', label='Cluster 5')
# Add centroids to the plot
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title("Clusters of Customers")
plt.xlabel("Anual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
