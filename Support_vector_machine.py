# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:41:38 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([[2], [3], [5], [7], [8]])
y = np.array([0, 0, 1, 1, 1])

# SVM implementation
from sklearn.svm import SVC
svm_classifier = SVC()
svm_classifier.fit(X, y)

# Making predictions
new_X = np.array([[4]])
prediction = svm_classifier.predict(new_X)
print("SVM Prediction:", prediction[0])
