"""
An Artificial Neural Network to carry out churn modeling which detects what
customers are most likley of leaving a company or cancelling a subscription etc
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

"""
---------------Data Pre-processing---------------
"""

# Importing dataset
dataset = pd.read_csv("..\Data\Artificial_Neural_Networks\Churn_Modelling.csv")
# Remove un-important columns such as customer ID, surname etc
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical data
# Label encoding the 'Gender' column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])

# One hot encoding on the 'Country' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
