"""
Using Thompson Sampling to choose the best ad for a product by analysing
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
import random

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
# List to keep track of how many times each ad is given a reward of 1 (clicked on)
num_of_rewards_1 = [0] * num_of_ads
# List to keep track of how many times each ad is given a reward of 0 (not clicked on)
num_of_rewards_0 = [0] * num_of_ads
# Total reward accumulated over all rounds
total_reward = 0

# ALGORITHM IMPLEMENTATION

# Iterate over all users (rows)
for n in range(N):
	ad = 0
	max_random_draw = 0
	# for each ad, take a random draw from its (beta) distribution
	for i in range(num_of_ads):
		random_beta_draw = random.betavariate(num_of_rewards_1[i] + 1, num_of_rewards_0[i] + 1)
		# Update variables keeping track of highest random draw and what ad to choose
		if random_beta_draw > max_random_draw:
			max_random_draw = random_beta_draw
			ad = i
	ads_selected.append(ad)
	# Reward for the ad increased if the user n (row) clicks on the ad (column)
    # Reward either 1 or 0, stored in dataset
	reward = dataset.values[n, ad]
    if reward == 1:
    	num_of_rewards_1[ad] += 1
    else:
    	num_of_rewards_0[ad] += 1
    total_reward += reward