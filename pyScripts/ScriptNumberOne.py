#%%#Imports
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import sys 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

#display all data:
def display_all(data):
    with pd.option_context("display.max_row", 100, "display.max_columns", 100):
        display(data)
################################################################################################################################################################
# Define the path you want to add
GETCWD = os.getcwd()

# Add the path to sys.path if it's not already there
if GETCWD not in sys.path:
    sys.path.append(GETCWD)
#add classes files from this dir 
# from SeeTheData import SeeTheData

################################################################################################################################################################ importing data
#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "/FGS_ML/ML_Project/data/diabetic_data.csv" )
PathToMap = os.path.join(GETCWD + "/../data/IDS_mapping.csv" )


#assing df
Maindf = pd.read_csv(PathToData)
Mapdf = pd.read_csv(PathToMap)

#sns + plt option and settings
sns.set_style("darkgrid")
plt.style.use("dark_background")

################################################################################################################################################################
df = Maindf
#Removing non-diabetes diagnosis should be before starting EDA
Subset_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]
df = Subset_df

################################################################################################################################################################
df.loc[df["readmitted"] == ">30" , "readmitted"] = "NO"
df= df.reset_index(drop= True)
################################################################################################################################################################
ColName= "readmitted"
df["categoricalValue"] = df[ColName]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df[ColName]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

def Ratio_cat_proportions(data):
    return data["categoricalValue"].value_counts() / len(data)
################################################################################################################################################################
train_set.to_csv("/home/labs/cssagi/barc/FGS_ML/ML_Project/data/train_set.csv")
train_set.to_csv(path_or_buf=None
