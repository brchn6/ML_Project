#Imports
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
GETCWD = os.getcwd()
import sys 

# Add the path to sys.path if it's not already there
if GETCWD not in sys.path:
    sys.path.append(os.path.join(GETCWD, "pyScripts"))


pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

from ScriptNumberOne import train_set
from ScriptNumberOne import Mapdf
################################################################################################################################################################
def drop_duplicates_fromDF(df,subset_col):
   NumberOf_patient_nbr_substract= df.shape[0] - df.drop_duplicates(subset=subset_col, keep="first").shape[0]
   df= df.drop_duplicates(subset=subset_col, keep="first")
   df= df.reset_index(drop=True)
   return df , NumberOf_patient_nbr_substract

#from now ion the train set is as follw:
# dim is [25727 rows x 51 columns]
#it was >>> train_set.shape (30419, 51)
train_set = drop_duplicates_fromDF(train_set,"patient_nbr") [0]
################################################################################################################################################################
#from now ion the train set is as follw: its because we drop akk the dead papule
# dim is [25468 rows x 51 columns]
#it was >>> [25727 rows x 51 columns]
discharge_disposition_id_DF = Mapdf[["discharge_disposition_id", "description.1"]]
discharge_disposition_id_DF = discharge_disposition_id_DF[discharge_disposition_id_DF['description.1'].str.contains('Hospice') | discharge_disposition_id_DF['description.1'].str.contains('Expired')]
train_set = train_set[~train_set['discharge_disposition_id'].isin(discharge_disposition_id_DF ['discharge_disposition_id'])]
################################################################################################################################################################
#Dropping columns with ALOT of missing values: base on the EDA 
# [25727 rows x 48 columns]
train_set = train_set.drop(['weight', 'medical_specialty', 'payer_code'], axis=1)
################################################################################################################################################################
#Checking for columns with just one kind of values,
#can adjust for more values (change the 2 in range function)
# [25468 rows x 43 columns]
# we drop ['acetohexamide', 'troglitazone', 'examide', 'citoglipton', 'metformin-rosiglitazone'] from 
# the train 
def some (df):
    unique_dict = {}
    col_list = []
    for col in df.columns:
        if np.dtype(df[col]) == 'object':
            for i in range(1,2):
                vals = pd.unique(df[col])
                unique_dict[col] = vals
                if len(vals) <= i :
                    col_list.append(col)
                    # print(f'column {col} has {i} unique values', unique_dict[col])
                    break
    return (col_list)

#Dropping cols with 1 kind of value:
train_set.drop(some(train_set), axis = 1)
################################################################################################################################################################

