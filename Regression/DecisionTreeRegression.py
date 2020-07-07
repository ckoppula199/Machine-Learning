"""
Decision tree regression model to predict the previous salary of an applicant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
