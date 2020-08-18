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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

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
    # remove negative word from stopwords
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    # Apply stemming and remove stopwords
    review = [ps.stem(word) for word in review if not word in set()]
    review = ' '.join(review)
    corpus.append(review)

"""
---------------Creating the Bag of Words Model---------------
"""

# Uses the most frequent n words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1]
# print(len(X[0]))

"""
---------------Training the Niave Bayes Classifier---------------
"""

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

"""
---------------Predicting Test Set Result---------------
"""
y_pred = classifier.predict(X_test)
# Convert vectors from horizontal to vertical and display side by side to compare
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

"""
--------------Making the Confusion Matrix and Evaluating the Model---------------
"""

# Gives confusion matrix C such that Cij is equal to the number of observations known to be in group i and predicted to be in group j.
cm = confusion_matrix(y_test, y_pred)
true_positive = cm[1][1]
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]

print("\nModel Evaluation")
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"Precision is {true_positive / (true_positive + false_positive)}")
print(f"Recall is {true_positive / (true_positive + false_negative)}")
print(f"F1 score is {f1_score(y_test, y_pred, average='binary')}")
