#%%
"""
This is the main file for the pyScripts package. It is the entry point for the package.
"""

#------------------------------Imports---------------------------------
from classes.GetTheData import PathToData, PathToMap
from classes.PreProssesingFunction import PreProcessingFunctions
import pandas as pd

#apply the pre-processing functions
def main():
    #------------------------------path to DataFrames----------------
    data = pd.read_csv(PathToData)
    map = pd.read_csv(PathToMap)
    #------------------------------Pre-processing----------------
    pre = PreProcessingFunctions()
    data = pre.remove_unwanted_columns_and_rows(data)
    train_set, test_set = pre.split_data(data, "readmitted")
    return data, map , train_set, test_set

if __name__ == "__main__":
    data, map , train_set, test_set = main()




# Set numerical columns
num_cols = ['num_medications', 'num_lab_procedures']
bin_cols = ['change', 'diabetesMed']
# Set categorical columns
cols = train_set.columns
label = 'readmitted'
cat_cols = [col for col in cols if col not in num_cols and col not in columns_to_drop and col not in label and col not in bin_cols]

