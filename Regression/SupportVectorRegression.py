import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScalar

"""
---------------Data Pre-processing---------------
"""

# Importing dataset
dataset = pd.read_csv("../Data/Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Convert y to a vector
y = t.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScalar()
sc_y = StandardScalar()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
