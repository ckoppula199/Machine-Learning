"""
Random forest regression model to predict the previous salary of an applicant.
Model performs poorly as the data only contains a single feature and this model
is better suited to multi-feature data sets.
This is an example of ensemble learning
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
---------------Training a Random Forest Regression Model---------------
"""
regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(X, y)

"""
---------------Predicting Results---------------
"""
print(regressor.predict([[6.5]]))

"""
---------------Visualising Random Forest Regression Results---------------
"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
