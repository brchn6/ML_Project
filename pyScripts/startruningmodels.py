#%%
# --------------------------- import the libraries and load the data --------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Load the data
from pyScripts.Run_pipeline import diabetes_prepared, diabetes_labels
# --------------------------- fit the models --------------------------- #

# Fit the models
linear_reg = LinearRegression()
linear_reg.fit(diabetes_prepared, diabetes_labels)

logistic_reg = LogisticRegression()
logistic_reg.fit(diabetes_prepared, diabetes_labels)


