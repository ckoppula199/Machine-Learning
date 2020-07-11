"""
Logistic Regression classification model to determine if someone will buy a new
SUV based on their age and salary.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""
---------------Data Pre-processing---------------
"""
# Data not split into training and test set due to small amount of data being
# processed

# Importing dataset
dataset = pd.read_csv("../Data/Classification/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 75% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
---------------Training the Model---------------
"""
# Training logistic regression model on training set
# Parameter C=1.0 represents the inverse regularisation term.
# Smaller C means stronger regulrisation
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

"""
---------------Predicting Single Result---------------
"""

single_prediction = classifier.predict([X_test[0, :]])
print(f"Single prediction is {single_prediction}")
print(f"Single prediction confidence [[0, 1]] is {classifier.predict_proba([X_test[0, :]])}")
print(f"Actual answer is {y_test[0]}")
