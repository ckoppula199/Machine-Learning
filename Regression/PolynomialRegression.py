"""
Polynomial regression model to predict the previous salary of an applicant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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


"""
---------------Training a Linear Regression Model---------------
"""

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


"""
---------------Training a Polynomial Regression Model---------------
"""
# Create new matrix of features including polynomials of features
# Change degree to see how the model changes
poly_regressor = PolynomialFeatures(degree=4)
# Matrix of features including polynomials of features upto degree n
X_poly = poly_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)


"""
---------------Visualising Linear Regression Results---------------
"""

plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title("Linear Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


"""
---------------Visualising Polynomial Regression Results---------------
"""

plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor_2.predict(X_poly), color='blue')
plt.title("Polynomial Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


"""
---------------Predicting Result with Linear Regression---------------
"""
# Predict result of 6.5 position level
print(linear_regressor.predict([[6.5]]))


"""
---------------Predicting Result with Polynomial Regression---------------
"""
# Predict result of 6.5 position level
# Cant use 6.5 as input, need to include all its polynomial features too
print(linear_regressor_2.predict(poly_regressor.fit_transform([[6.5]])))
