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
import os
import sys

#-----------------------set the working directory--------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if sys.path[0] != parent_dir:
    sys.path.insert(0, parent_dir)
# ---------- add the path to the config file --------------------------
from prepdir import Config
Config.EnvPrepa_main()
sys.path
#---load the data---

from RunPipe import X_train, X_test, y_train, y_test
#%% 

#---create the model---
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

# Train the model on training data
rf.fit(X_train, y_train)

from RunPipe import X_train, X_test, y_train, y_test
