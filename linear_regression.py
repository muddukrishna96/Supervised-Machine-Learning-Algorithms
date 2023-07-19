# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:41:39 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Linear Regression implementation
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# Making predictions
new_X = np.array([6]).reshape(-1, 1)
prediction = lr.predict(new_X)
print("Linear Regression Prediction:", prediction[0])
