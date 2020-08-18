"""
Program to determine whether a review of a restaurant is positive or negative
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

"""
---------------Data Pre-Processing---------------
"""

# .tsv file is seperated by tabs and quoting=3 means to ignore all quotes
dataset = pd.read_csv('../Data/Natural_Language_Processing/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
# Need to execute nltk.download('stopwords') if not previously done
corpus = []
for i  in range(len(dataset)):
    # remove punctuation
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # convert to lowercase
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # Apply stemming and remove stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
