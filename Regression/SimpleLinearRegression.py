"""
Simple Linear Regression used to predict salary of an employee based on an
employees number of years of experience
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
---------------Data Pre-processing---------------
"""

# Importing dataset
dataset = pd.read_csv("../Data/Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


"""
---------------Training the Model---------------
"""

# Training simple linear regression model on training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)


"""
---------------Predicting Test Set Results---------------
"""

# Predicting the test set results and finding sum of difference
prediction = regressor.predict(X_test)
print(prediction)
sum_difference = 0

# Prints out sum difference between predicted and actaul values
"""
for actual, predicted in zip(y_test, prediction):
    sum_difference += abs(actual - predicted)

print(f"Sum difference between predicted and actual values is {sum_difference}")
"""


"""
---------------Visualising the Data and Results---------------
"""

# Visualising Training set data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the test set data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
