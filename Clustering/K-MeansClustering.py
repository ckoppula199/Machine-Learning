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
plt.plot(range(1, 11), wcss)
plt.title("Elboe Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum of Squares")
plt.show()

# Graph shows that 5 clusters is the optimal amount
