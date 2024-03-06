'''
This script is used to run Rendom Forest model on the data

Author: Barc
Date: 2024-03-06
'''

#---import the required libraries---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#---load the data---
from RunPipe import X_train, X_test, y_train, y_test

#---create the model---
model = RandomForestClassifier(n_estimators=100, random_state=0)
