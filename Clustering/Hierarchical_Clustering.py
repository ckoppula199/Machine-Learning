"""
Hierarchical Clustering model to determine patterns in customer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

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
