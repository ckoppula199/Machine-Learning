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
from sklearn.model_selection import train_test_split]
from sklearn.preprocessing import StandardScaler

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

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Apply feature scaling to the data
sc = StandardScaler
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
