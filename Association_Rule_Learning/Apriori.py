"""
Using the Apriori Association Rule Learning algorithm to find out patterns in
items bought by customers at a supermarket at to determine what items a person
is likely to buy together.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

"""
---------------Data Pre-processing---------------
"""

# dataset has no headers so want to take into account first row
# dataset contains a weeks worth of shopping transaction history
dataset = pd.read_csv('../Data/Association_Rule_Learning/Market_Basket_Optimisation.csv', header=None)

# apyori module requires data in a list format and they are strings
transactions = []
for i in range(len(dataset)):
    # At most 20 items purchased at once
    transactions.append([str(dataset.values[i, j]) for j in range(20)])


"""
---------------Training Apriori Model---------------
"""

# apriori function returns the association rules
# considers items that appears at least 3 times a day (min_support=21/len(dataset))
# min_confidence found using trail and error
# min_lift=3 means only recommend rules where a shopper is 3 times more likely to buy item A
# given that they bought item B
apriori_rules = apriori(transactions=transactions, min_support=(21/len(dataset)), min_confidence=0.2, min_lift=3, min_length=2, max_length=2)


"""
---------------Visualising the Rules---------------
"""
results = list(apriori_rules)
print(results, end='\n\n\n')

# Function to neatly organise the results of the association rules

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

dataframe_results = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Supports', 'Confidences', 'Lifts'])
print(dataframe_results.nlargest(n=10, columns='Lifts'), end='\n\n\n')
