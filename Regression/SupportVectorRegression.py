"""
Support vector regression model to predict the previous salary of an applicant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

"""
---------------Data Pre-processing---------------
"""
# Data not split into training and test set due to small amount of data being
# processed

# Importing dataset
dataset = pd.read_csv("../Data/Regression/Position_Salaries.csv")
# Exclude first column
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Convert y to a vector
y = y.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

"""
---------------Training a Support Vecotr Regression Model---------------
"""
# Uses Radial Basis Function
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

"""
---------------Predicting Results---------------
"""
value_to_predict = 6.5
scaled_value = sc_X.transform([[value_to_predict]])
print(sc_y.inverse_transform(regressor.predict(scaled_value)))

"""
---------------Visualising Support Vector Regression Results---------------
"""
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color='blue')
plt.title("Support Vector Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title("Support Vector Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
