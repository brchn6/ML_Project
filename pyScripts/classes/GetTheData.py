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
#------------------------------get the root dir "ML_Project"----------------

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
while not ROOT_DIR.endswith("ML_Project"):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

#------------------------------add the root dir to the sys path----------------
# logic to make the root folder as the root dir
if ROOT_DIR.endswith("ML_Project"):
    sys.path.append(ROOT_DIR)
else:
    print("Error: You are not in the root directory of the project")

#------------------------------Path to the data----------------
PathToData = os.path.join(ROOT_DIR, "Data", "diabetic_data.csv")
PathToMap = os.path.join(ROOT_DIR, "Data", "IDs_mapping.csv")


