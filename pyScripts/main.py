#%%
""""
This is the fucking main file, all file should be called from here
"""
# ---------------------------- Imports -------------------------------
#from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from AddRootDirectoriesToSysPath import AddRootDirectoriesToSys #importing the function that add all path in the dir to sys
AddRootDirectoriesToSys() #implement this function
import pandas as pd
from deadendscript.feature_importance_gans_4_numeric import *
from DefPipeLineClasses import DropColumns

# --------------------- Set up the X and y Train and Test value -----------------------
#call the X_train from csv name copula_train_set.csv
X_train = pd.read_csv('../data/copula_train_set_300_epochs_4_numeric.csv')
#extract the lables
y_train = X_train['readmitted']
#remove labales from X_train
X_train = X_train.drop('readmitted', axis=1)
#drop the columns:
X_train = DropColumns(cols_to_drop).fit_transform(X_train)
#Change 2 columns to float64 (for pipline purposes) (54428 rows × 30 columns)
X_train = convert_to_float64(X_train, cols_to_change)
#fit the data: (54428x119)
X_train_np = col_processor.fit_transform(X_train)


import os
if not os.getcwd() == "/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts":
    os.chdir("/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts")

dir_to_save= "./BarModels"

#save the X_train_np matrix to a file
import numpy as np
np.save(dir_to_save + "/X_train_np", X_train_np)
#save the y_train matrix to a file
np.save(dir_to_save + "/y_train", y_train)
