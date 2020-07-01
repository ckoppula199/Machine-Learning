"""
Multivariate Linear Regression used to predict the profit of a company depending
on the amount spent on R&D, Administration, Marketing spend and State.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

"""
---------------Data Pre-processing---------------
"""

# Importing dataset
dataset = pd.read_csv("../Data/Regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode catagorical data using one hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

"""
---------------Training the Model---------------
"""
# Class automatically handles dummy variable trap and choose the most
# statistically significant features to use in the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""
---------------Predicting Test Set Results---------------
"""

# Predicting values of the test set
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) # 2 dp values
# Convert vectors from horizontal to vertical and display side by side to compare
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))
# Prints out sum difference between predicted and actaul values
sum_difference = 0
for actual, predicted in zip(y_test, y_pred):
    sum_difference += abs(actual - predicted)

print(f"Sum difference between predicted and actual values is {sum_difference}")
