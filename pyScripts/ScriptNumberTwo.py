#Imports
#%%
#%%
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
    sys.path.append(os.path.join(GETCWD, "FGS_ML/ML_Project/pyScripts"))

# /home/labs/cssagi/barc/FGS_ML/ML_Project/data/diabetic_data.csv
pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

#importnig the trainDS from script number 1
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
#from now ion the train set is as follw: its because we drop akk the dead people
# dim is [25468 rows x 51 columns]
#it was >>> [25727 rows x 51 columns]
discharge_disposition_id_DF = Mapdf[["discharge_disposition_id", "description.1"]]
discharge_disposition_id_DF = discharge_disposition_id_DF[discharge_disposition_id_DF['description.1'].str.contains('Hospice') | discharge_disposition_id_DF['description.1'].str.contains('Expired')]
train_set = train_set[~train_set['discharge_disposition_id'].isin(discharge_disposition_id_DF ['discharge_disposition_id'])]
################################################################################################################################################################
#Dropping columns with ALOT of missing values: base on the EDA 
# [25468 rows x 48 columns]
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
train_set = train_set.drop(some(train_set), axis = 1)
################################################################################################################################################################
#Grouping diseases by the ID:
diseases = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes','Diabetes Uncontrolled', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
ids = ids = [
    (list(range(390, 460)) + [785]),
    (list(range(460, 520)) + [786]),
    (list(range(520, 580)) + [787]),
    ([250.00, 250.01]),
    np.round(np.arange(250.02, 251.00, 0.01), 2).tolist(),
    (list(range(800, 1000))),
    (list(range(710, 740))),
    (list(range(580, 630)) + [788]),
    (list(range(140, 240))),  
    (list(range(790, 800)) + 
    list(range(240, 250)) + 
    list(range(251, 280)) + 
    list(range(680, 710)) + 
    list(range(780, 785)) + 
    list(range(290, 320)) + 
    list(range(280, 290)) + 
    list(range(320, 360)) + 
    list(range(630, 680)) + 
    list(range(360, 390)) + 
    list(range(740, 760)) +
    list(range(1,140)))
]
diag_columns = ['diag_1', 'diag_2', 'diag_3']

len(ids)
len(diseases)

# Function to convert values in the 'Value' column based on the ranges
def convert_values(value):
    if value == '?' or value == '789':
        return diseases[9]
    try:
        numeric_value = float(value)
    except ValueError:
        return diseases[9]  # Skip non-numeric values

    for id, disease in zip(ids, diseases):
        if numeric_value in id:
            return disease  # Replace with the string of your choosing
    return value

for col in diag_columns:
    train_set[col] = train_set[col].apply(convert_values)

train_set_melted = pd.melt(train_set.loc[:,'diag_1':'diag_3'])
plt.figure(figsize=(15,8))

ax = sns.countplot(x='value', hue='variable', data=train_set_melted)
plt.show()

train_set["diag_1"].unique()

#Ended up with 10 categories for each diag column,
#Diabetes is plitted to uncontrolled dibetes and diabetes
################################################################################################################################################################
# %%
#Creating 6 features for A1Cresult column which describes the results of the HbA1c test
#if one was done on the patient.
cond1 = train_set['A1Cresult'] == 'None'
cond2 = (train_set['A1Cresult'] == 'Norm')
cond4 = (train_set['A1Cresult'] == '>7') & (train_set['change'] == "Ch")
cond3 = (train_set['A1Cresult'] == '>7') & (train_set['change'] == "No")
cond5 = (train_set['A1Cresult'] == '>8') & (train_set['change'] == "No")
cond6 = (train_set['A1Cresult'] == '>8') & (train_set['change'] == "Ch")

# %%
train_set.loc[cond1, 'A1Cresult'] = 'No HbA1c test performed'
train_set.loc[cond2, 'A1Cresult'] = 'HbA1c in normal range'
train_set.loc[cond3, 'A1Cresult'] = 'HbA1c greater than 7%, but no med change'
train_set.loc[cond4, 'A1Cresult'] = 'HbA1c greater than 7%, with med change'
train_set.loc[cond5, 'A1Cresult'] = 'HbA1c greater than 8%, but no med change'
train_set.loc[cond6, 'A1Cresult'] = 'HbA1c greater than 8%, with med change'
################################################################################################################################################################
# %%
#Regrouping columns from Mapdf:
train_set['admission_type_id'] = train_set['admission_type_id'].replace([8,6],5).replace([7],6)

train_set['discharge_disposition_id'] = train_set['discharge_disposition_id'].replace([list(range(3,6)) + [10,15,9,23,24,22]+ list(range(27,31))],2).replace([6,8],3).replace(7,4).replace(12,5).replace([16,17],6).replace([25,26,18],7)

train_set['admission_source_id'] = train_set['admission_source_id'].replace([2,3],1).replace([25,22,18,19,10,5,6,7,4],2).replace(8,3).replace([19,20,17,15,9],4).replace([23,24,11,12,13,14],5)

# %%
#Regrouping Age column:
train_set['age'] = train_set['age'].replace(['[0-10)'],1).replace(['[10-20)'],2).replace(['[20-30)','[30-40)','[40-50)','[50-60)','[60-70)'],3).replace(['[70-80)','[80-90)','[90-100)'],4)
# %%
################################################################################################################################################################
#Convering continious values to categorial: 
def replaceNumEmergency(value):
    if value == 0:
        return str(value)
    elif (value > 0) & (value < 5):
        return '<5'
    else:
        return '>=5'

train_set['number_emergency'] = train_set['number_emergency'].apply(replaceNumEmergency)

def timeInHosp(value):
    if (value >= 1) & (value <= 4):
        return '1-4'
    elif (value > 4) & (value <= 8):
        return '5-8'
    else:
        return '>8'

train_set['time_in_hospital'] = train_set['time_in_hospital'].apply(timeInHosp)

def numProcedures(value):
    if value == 0:
        return str(value)
    elif (value >= 1) & (value <= 3):
        return '1-3'
    else:
        return '4-6'
    
train_set['num_procedures'] = train_set['num_procedures'].apply(numProcedures)   

def inPatiant(value):
    if value == 0:
        return str(value)
    elif (value >= 1) & (value <= 5):
        return '1-5'
    else:
        return '>5'

train_set['number_inpatient'] = train_set['number_inpatient'].apply(inPatiant)     

def numDiag(value):
    if (value >= 1) & (value <= 4):
        return '1-4'
    elif (value > 4) & (value <= 8):
        return '5-8'
    else:
        return '>=9'

train_set['number_diagnoses'] = train_set['number_diagnoses'].apply(numDiag)     

# %%
