#%%
"""
this script is for the first step in the project
1- Importing the data
2- Splitting the data into train and test set with the help of StratifiedShuffleSplit
"""
#------------------------------Imports---------------------------------
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import sys 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#------------------------------settings--------------------------------

pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

#set seed
np.random.seed(42)

#get the root dir "'y:\\barc\\FGS_ML\\ML_Project\
ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
ROOT_DIR = os.path.join(ROOT_DIR, "../")

#add the script "AddRootDirectoriesToSysPath" and implament it
sys.path.append(ROOT_DIR)
from AddRootDirectoriesToSysPath import add_directories_to_sys
from sklearn.preprocessing import LabelEncoder
add_directories_to_sys(ROOT_DIR)

#sns + plt option and settings
sns.set_style("darkgrid")
plt.style.use("dark_background")

# Define the path you want to add
GETCWD = os.getcwd()

#------------------------------get the data file ----------------
if os.path.basename(GETCWD) == "pyScripts":
    PathToData = os.path.join(GETCWD + "/../data/diabetic_data.csv" )
    PathToMap = os.path.join(GETCWD + "/../data/IDS_mapping.csv" )
#adding a logic to the path of the data file do i could work from any dir
elif os.path.basename(GETCWD) == "barc":
    PathToData = os.path.join(GETCWD + "/FGS_ML/ML_Project/data/diabetic_data.csv" )
    PathToMap = os.path.join(GETCWD + "/FGS_ML/ML_Project/data/IDS_mapping.csv" )
elif os.path.basename(GETCWD) == "ML_Project":
    PathToData = os.path.join(GETCWD + "/data/diabetic_data.csv" )
    PathToMap = os.path.join(GETCWD + "/data/IDS_mapping.csv" )
elif os.path.basename(GETCWD) == "FGS_ML":
    PathToData = os.path.join(GETCWD + "/data/diabetic_data.csv" )
    PathToMap = os.path.join(GETCWD + "/data/IDS_mapping.csv" )

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

def split_data_group(df, ColName):
    split = StratifiedGroupKFold(n_splits=5)
    for train_index, test_index in split.split(df, df[ColName], groups=df['patient_nbr']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
    return train_set, test_set

#---------------------------------Main--------------------------------
#main function to run the script
def prepare_data_main(method = None):
    ColName= "readmitted"
    Maindf = pd.read_csv(PathToData)
    Mapdf = pd.read_csv(PathToMap)
    Maindf = remove_unwanted_columns_and_rows(Maindf)
    if method == "group":
        train_set, test_set = split_data_group(Maindf, ColName)
    elif method == "normal":
        train_set, test_set = split_data(Maindf, ColName)
    return train_set, test_set ,Mapdf


