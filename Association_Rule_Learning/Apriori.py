"""
Using the Apriori Association Rule Learning algorithm to find out patterns in
items bought by customers at a supermarket at to determine what items a person
is likely to buy together.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
---------------Data Pre-processing---------------
"""

# dataset has no headers so want to take into account first row

dataset = pd.read_csv('../Data/Association_Rule_Learning/Market_Basket_Optimisation.csv', header=None)

# apyori module requires data in a list format and they are strings
transactions = []
for i in range(len(dataset)):
    # At most 20 items purchased at once
    transactions.append([str(dataset.values[i, j]) for j in range(20)])


"""
---------------Training Apriori Model---------------
"""
