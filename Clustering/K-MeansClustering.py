"""
K-Means Clustering model to determine patterns in customer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
