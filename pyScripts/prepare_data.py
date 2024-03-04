#%%
"""
this script is for the first step in the project
1- Importing the data
2- Splitting the data into train and test set with the help of StratifiedShuffleSplit
3- Saving the train set into a csv file
"""
#Imports
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
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

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
add_directories_to_sys(ROOT_DIR)

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
df = df.reset_index(drop= True)
df['readmitted'] = le.fit_transform(df[['readmitted']]) 
df['readmitted'] = df['readmitted'].astype('object')
##########################################################################################
#def to_categorical(data):
#    object_columns = data.select_dtypes(include=['object']).columns
#    if len(object_columns) == 0:
#        return data
#    else:
#        data[object_columns] = data[object_columns].astype('category')  
#df1 = to_categorical(df)
################################################################################################################################################################
ColName= "readmitted"

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df[ColName]):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

def Ratio_cat_proportions(data):
    return data["categoricalValue"].value_counts() / len(data)
################################################################################################################################################################
print(train_set.columns)

#train_label = train_set['readmitted']
#train_set = train_set.drop(columns=['readmitted'])
#############################
#if we run this line : the train set will be difffrent from the one in the main file
# train_set.to_csv(os.getcwd() + "/../data/train_set_test.csv", index=False)
# train_set.to_csv(path_or_buf=None)

#####
#train_set.to_csv('saar.csv')
