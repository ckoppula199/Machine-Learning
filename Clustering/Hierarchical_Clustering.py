"""
Hierarchical Clustering model to determine patterns in customer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

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
---------------Using Dendrogram to Find Optimal Number of Clusters---------------
"""
# method='ward' means we are using minimum variance as the clustering technique
# Graph shows both either 3 or 5 clusters are a good choice
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers (Observation points)")
plt.ylabel("Euclidean Distance")
plt.show()


"""
---------------Training the Hierarchical Clustering Model on the Dataset---------------
"""
# affinty is the type of distance used to measure distance between clusters
# linkage='ward' means minimum variance used as the clustering technique
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred = hc.fit_predict(X)


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

plt.title("Clusters of Customers")
plt.xlabel("Anual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
