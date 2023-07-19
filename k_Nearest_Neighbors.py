# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:41:39 2023

@author: mgalaval
"""

import numpy as np

# Sample data
X = np.array([[2], [3], [5], [7], [8]])
y = np.array([0, 0, 1, 1, 1])

# k-NN implementation
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X, y)

# Making predictions
new_X = np.array([[4]])
prediction = knn_classifier.predict(new_X)
print("k-NN Prediction:", prediction[0])
