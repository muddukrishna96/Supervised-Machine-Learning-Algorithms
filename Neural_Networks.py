# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:46:00 2023

@author: mgalaval
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Neural Network implementation
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1)

# Making predictions
new_X = np.array([[6]])
prediction = model.predict(new_X)
print("Neural Network Prediction:", int(prediction[0][0] + 0.5))
