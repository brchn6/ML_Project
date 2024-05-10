#%%
""""
This is the fucking main file, all file should be called from here
"""
# ---------------------------- Imports -------------------------------
#from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # This line in for sitting the working dir to be the same as the script loc
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

cols_to_drop = cols_to_drop
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

#%%
# --------------------- Makeing DMatrix file for XGBOOST -----------------------
"""

X_train = pd.read_csv('../data/copula_train_set_300_epochs_4_numeric.csv')
X_test = pd.read_csv('../data/copula_gans_4_numeric_TestSet.csv')

y_train = X_train['readmitted']
y_test = X_test['readmitted']

X_train = X_train.drop('readmitted', axis=1)
X_test = X_test.drop('readmitted', axis=1)

X_train = DropColumns(cols_to_drop).fit_transform(X_train)
X_test = DropColumns(cols_to_drop).fit_transform(X_test)

X_train = convert_to_float64(X_train, cols_to_change)
X_test = convert_to_float64(X_test, cols_to_change)

X_train_np = col_processor.fit_transform(X_train)

names_train = col_processor.get_feature_names_out()

feature_names_train = [name[12:] for name in names_train]

X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train_np, columns=feature_names_train)

#do the same for the test set:
X_test_np = col_processor.fit_transform(X_test)

names_test = col_processor.get_feature_names_out()

feature_names_test = [name[12:] for name in names_test]

X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test_np, columns=feature_names_test)


#export X_train_df and X_test_df and y_train and y_test to csv files:
X_train_df.to_csv('GuyTrain/X_train_df.csv', index=False)
X_test_df.to_csv('GuyTrain/X_test_df.csv', index=False)
y_train.to_csv('GuyTrain/y_train.csv', index=False)
y_test.to_csv('GuyTrain/y_test.csv', index=False)
"""
#%%
#Test X_traim_df on models:

X_train_df = pd.read_csv('GuyTrain/X_train_df.csv')
X_test_df = pd.read_csv('GuyTrain/X_test_df.csv')
y_train = pd.read_csv('GuyTrain/y_train.csv')
y_test = pd.read_csv('GuyTrain/y_test.csv')

def convert_to_float64(dataframe, columns):
    dataframe[columns] = dataframe[columns].astype('float64')
    return dataframe

cols_to_change = ['number_emergency', 'number_outpatient', 'readmitted']

X_train = convert_to_float64(X_train, cols_to_change)

columns = ['diag_3_365.44', 'repaglinide_Down']

def removeRogueColumns(df):
        df.drop(columns, axis=1, inplace=True)
        return df

X_train = X_train.drop('readmitted', axis=1)

X_train_df = removeRogueColumns(X_train_df)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_pipe = make_impipe(col_processor, xgb_clf)

xgb_cv_cv = cross_validate(cv_pipe, X_train, y_train, cv=cv, scoring='neg_log_loss')

xgb_cv_np = cross_validate(xgb_clf, X_train_np, y_train, cv=cv, scoring='neg_log_loss')

xgb_cv_df = cross_validate(xgb_clf, X_train_df, y_train, cv=cv, scoring='neg_log_loss')


# %%
