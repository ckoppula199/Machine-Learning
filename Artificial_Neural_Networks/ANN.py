"""
An Artificial Neural Network to carry out churn modeling which detects what
customers are most likley of leaving a company or cancelling a subscription etc
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

"""
---------------Data Pre-processing---------------
"""

# Importing dataset
dataset = pd.read_csv("..\Data\Artificial_Neural_Networks\Churn_Modelling.csv")
# Remove un-important columns such as customer ID, surname etc
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical data
# Label encoding the 'Gender' column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])

# One hot encoding on the 'Country' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 80% of data used for training and fixed random seed so that when re-ran same
# data in the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Apply feature scaling to the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(X_test.shape)

"""
Initialising the Aritificial Neural Network
"""

ann = tf.keras.models.Sequential()
# Add input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(X_train.shape[1],)))
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""
Training the Aritificial Neural Network
"""
# Compileing the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

"""
Making Predictions and Evaluating the Model
"""
# Predicting a single result
# Predict Geography: France, Credit Score: 600, Gender: Male, Age: 40, Tenure: 3 years,
# Balance: 60000, Number of Products: 2, Has Credit Card?: Yes, Is Active Member?: Yes,
# Estimated Salary: 50000
leave_prob = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
if leave_prob > 0.5:
    print("Customer is likely to leave the bank")
else:
    print("Customer is unlikely to leave the bank")

# Use model on the Test set
y_pred = ann.predict(X_test)
# Boolean list of whether or not customer is likely to leave
y_pred = (y_pred > 0.5)
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
