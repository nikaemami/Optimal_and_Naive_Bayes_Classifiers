#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv("/Users/Nika/Desktop/Iris.csv")
X = dataset.iloc[:, [0, 1, 2, 3]].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

nb = GaussianNB()
nb.fit(X_train, Y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
Y_pred = nb.predict(X_test)
print("confusion matrix = ", confusion_matrix(Y_test, Y_pred))
print("accuracy = ", accuracy_score(Y_test, Y_pred))


