#%%
'''
This script is used to run Rendom Forest model on the data

Author: Barc
Date: 2024-03-06
'''

#---import the required libraries---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


#---load the data---
from RunPipe import X_train, X_test, y_train, y_test

#---create the model---
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

# Train the model on training data
rf.fit(X_train, y_train)