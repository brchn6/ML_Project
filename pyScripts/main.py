
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
#call the DFS from the path
X_train = pd.read_csv('../data/copula_train_set_300_epochs_4_numeric.csv')
X_test = pd.read_csv('../data/copula_gans_4_numeric_TestSet.csv')
#function time
def FunctionToGet_Y_lables_fromTrainAndTest(X_train, X_test):
    # make the lable var as y_train and y_test
    y_train = X_train['readmitted']
    y_test = X_test['readmitted']
    #remove the lables from the train and test set
    X_train = X_train.drop('readmitted', axis=1)
    X_test = X_test.drop('readmitted', axis=1)
    #apply the drop columns function to the train and test set
    X_train = DropColumns(cols_to_drop).fit_transform(X_train)
    X_test = DropColumns(cols_to_drop).fit_transform(X_test)
    #convert the columns to float64
    X_train = convert_to_float64(X_train, cols_to_change)
    X_test = convert_to_float64(X_test, cols_to_change)
    #apply the col processor to the train and test set
    X_train_np = col_processor.fit_transform(X_train)
    X_test_np = col_processor.fit_transform(X_test)
    #return all
    return X_train_np, X_test_np, y_train, y_test
# call the function
X_train_np, X_test_np, y_train, y_test = FunctionToGet_Y_lables_fromTrainAndTest(X_train, X_test)

#%%
# from BarModels.GetXYstes import save_data
# save_data(X_train_np, y_train, X_test_np, y_test, dir_to_save= "./BarModels")
