# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:41:38 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Random Forest implementation
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)

# Making predictions
new_X = np.array([[6]])
prediction = rf_classifier.predict(new_X)
print("Random Forest Prediction:", prediction[0])
