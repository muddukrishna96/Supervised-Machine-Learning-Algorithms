# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:46:00 2023

@author: mgalaval
"""
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# SVR implementation
from sklearn.svm import SVR
svr_regressor = SVR()
svr_regressor.fit(X, y)

# Making predictions
new_X = np.array([6]).reshape(-1, 1)
prediction = svr_regressor.predict(new_X)
print("SVR Prediction:", prediction[0])
