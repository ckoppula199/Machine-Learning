"""
Using Upper Confidence Bound to choose the best ad for a product by analysing
the click through rate of each add.
"""

"""
In the dataset each row corresponds to a user, each column is an add and each cell
is 1 if the user would have clicked on the ad and 0 otherwise. In reality this method
would be applied in real time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

"""
---------------Data Pre-processing---------------
"""

# All that's needed for this is to import the dataset
dataset = pd.read_csv('../Data/Reinforcement_Learning/Ads_CTR_Optimisation.csv')


"""
---------------Implementing the UCB Algorithm---------------
"""

# VARIABLE INITIALISATION

# Total number of users ads are shown to (number of rounds)
# Dataset contains 10000 rows
N = 10000
num_of_ads = 10
# List of selected ads over the rounds
ads_selected = []
# List to keep track of how many times each ad is advertised
num_of_selections = [0] * num_of_ads
# List to keep track of each ads rewards (num of times it was clicked on)
sums_of_rewards = [0] * num_of_ads
# Total reward accumulated over all rounds
total_reward = 0

# ALGORITHM IMPLEMENTATION

# Iterate over all users (rows)
for n in range(N):
    ad = 0
    max_upper_bound = 0
    # Recalculate upper confidence bound for each add based on new user data (iterate over cols)
    for i in range(num_of_ads):
        if num_of_selections[i] > 0:
            # Compute average reward of ad i so far (num of times ad was clicked on / number of times ad was shown)
            average_reward = sums_of_rewards[i] / num_of_selections[i]
            # Compute confidence interval
            delta_i = math.sqrt(3/2 * math.log(n + 1) / num_of_selections[i])
            # Compute new upper bound of the ad
            upper_bound = average_reward + delta_i
        else:
            # Deals with case where ad hasn't been seen before to prevent divide by 0 error
            upper_bound = math.inf
        # Update variables keeping track of highest upper bound and what ad to choose
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    # Number of times ad was shown is incremented
    num_of_selections[ad] += 1
    # Reward for the ad increased if the user n (row) clicks on the ad (column)
    # Reward either 1 or 0, stored in dataset
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward


"""
---------------Visualising the Results---------------
"""

# Histogram plots for each of the ads how much it was selected
plt.hist(ads_selected)
plt.title("Histogram of Ad Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
