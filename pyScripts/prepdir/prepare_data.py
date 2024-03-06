#%%
"""
this script is for the first step in the project
1- Importing the data
2- Splitting the data into train and test set with the help of StratifiedShuffleSplit
"""
#------------------------------Imports---------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import sys
import os
le = LabelEncoder()

#-----------------------set the working directory--------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if sys.path[0] != parent_dir:
    sys.path.insert(0, parent_dir)
# ---------- add the path to the config file --------------------------
from prepdir import Config
from Config import *
Config.EnvPrepa_main()

#------------------------------get the path from the EnvPrepa.py file ----------------

PathToData, PathToMap = define_data_paths()

#------------------------------settings---------------------------------
pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

#set seed
np.random.seed(42)

#sns + plt option and settings
sns.set_style("darkgrid")
plt.style.use("dark_background")

# ---------------------------------Functions--------------------------------

def remove_unwanted_columns_and_rows(df):
    Subset_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]
    df = Subset_df
    df.loc[df["readmitted"] == ">30" , "readmitted"] = "NO"
    df = df.reset_index(drop= True)
    df['readmitted'] = le.fit_transform(df[['readmitted']]) 
    return df

def split_data(df, ColName):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[ColName]):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    return train_set, test_set

#---------------------------------Main--------------------------------
#main function to run the script
def prepare_data_main():
    ColName= "readmitted"
    Maindf = pd.read_csv(PathToData)
    Mapdf = pd.read_csv(PathToMap)
    Maindf = remove_unwanted_columns_and_rows(Maindf)
    train_set, test_set = split_data(Maindf, ColName)
    return train_set, test_set ,Mapdf
