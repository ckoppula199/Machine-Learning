"""
Program that uses Principal Component Analysis to reduce the dimensionality of a
dataset before applying logisitc regression to determine which type of wine
a person likes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


"""
---------------Data Pre-processing---------------
"""
# Importing dataset
dataset = pd.read_csv('../Data/Dimensionality_Reduction/Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].valuus

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
---------------Applying Principal Componenet Analysis---------------
"""
# reduce to 2 dimensions in order to visualise the results
pca = PCA(n_components=2)


"""
---------------Training Logisitic Regression Classifier---------------
"""
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)


"""
---------------Making Confusion Matrix---------------
"""
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
true_positive = cm[1][1]
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]

print("\nModel Evaluation")
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"Precision is {true_positive / (true_positive + false_positive)}")
print(f"Recall is {true_positive / (true_positive + false_negative)}")
print(f"F1 score is {f1_score(y_test, y_pred, average='binary')}")
