# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:41:38 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([[2], [3], [5], [7], [8]])
y = np.array([0, 0, 1, 1, 1])

# Logistic Regression implementation
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Making predictions
new_X = np.array([[4]])
prediction = log_reg.predict(new_X)
print("Logistic Regression Prediction:", prediction[0])
