#%%
""""
This is the fucking main file, all file should be called from here
"""
# ---------------------------- Imports -------------------------------
from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
from AddRootDirectoriesToSysPath import AddRootDirectoriesToSys #importing the funcrion htat add all path in the dir to sys
AddRootDirectoriesToSys() #implament this function
import pandas as pd


#%%
X_train = pd.read_csv('../data/copula_train_set.csv')
y_train = X_train['readmitted']
X_train = X_train.drop('readmitted', axis=1)

X_train = col_processor.fit_transform(X_train)
X_train

