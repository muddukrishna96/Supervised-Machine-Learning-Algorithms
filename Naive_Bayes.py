# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:46:00 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Naive Bayes implementation
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X, y)

# Making predictions
new_X = np.array([[6]])
prediction = nb_classifier.predict(new_X)
print("Naive Bayes Prediction:", prediction[0])
