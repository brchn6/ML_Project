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
from RunPipe import X_test

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

# #%%
# from BarModels.GetXYstes import save_data
# save_data(X_train_np, y_train)

