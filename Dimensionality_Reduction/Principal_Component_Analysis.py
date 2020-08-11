"""
Program that uses Principal Component Analysis to reduce the dimensionality of a
dataset before applying logisitc regression to determine which type of wine
a person likes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

"""
---------------Data Pre-processing---------------
"""
# Importing dataset
dataset = pd.read_csv('../Data/Dimensionality_Reduction/Wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
---------------Applying Principal Componenet Analysis---------------
"""
# reduce to 2 dimensions in order to visualise the results
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


"""
---------------Training Logisitic Regression Classifier---------------
"""
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


"""
---------------Making Confusion Matrix---------------
"""
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation")
print(f"Accuracy is {accuracy_score(y_test, y_pred)}")
print(f"F1 score is {f1_score(y_test, y_pred, average='weighted')}")


"""
---------------Visualising Results---------------
"""
# Training set
# Code for visualisation is not my own and has been taken from a tutorial on
# visualising data from PCA
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Test set
# Code for visualisation is not my own and has been taken from a tutorial on
# visualising data from PCA
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Testing Set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
