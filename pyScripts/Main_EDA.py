#%%
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


pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

#display all data:
def display_all(data):
    with pd.option_context("display.max_row", 100, "display.max_columns", 100):
        display(data)
#%%
# Define the path you want to add
GETCWD = os.getcwd()

# Add the path to sys.path if it's not already there
if GETCWD not in sys.path:
    sys.path.append(GETCWD)
#add classes files from this dir 

#%% importing data
#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "\\data/diabetic_data.csv" )
PathToMap = os.path.join(GETCWD + "\\data/IDS_mapping.csv" )
#assing df
Maindf = pd.read_csv(PathToData)
Mapdf = pd.read_csv(PathToMap)

#sns + plt option and settings
sns.set_style("darkgrid")
plt.style.use("dark_background")

#%% SeeTheData script OOP will be use in the future
# a= SeeTheData(df)
# a.Subsetting()
# # a.Display()
# a.CountPlotOfObjectColumns()
# a.HistPlotOfNumericColumns()
#%%
df = Maindf
#Removing non-diabetes diagnosis should be before starting EDA
Subset_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]
df = Subset_df

#%%

df.loc[df["readmitted"] == ">30" , "readmitted"] = "NO"
df= df.reset_index(drop= True)
#%%
ColName= "readmitted"
df["categoricalValue"] = df[ColName]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df[ColName]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

def Ratio_cat_proportions(data):
    return data["categoricalValue"].value_counts() / len(data)

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#%%
compare_props = pd.DataFrame({
    "Overall": Ratio_cat_proportions(df),
    "Stratified": Ratio_cat_proportions(strat_test_set),
    "Random": Ratio_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
display(compare_props)

f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)
sns.barplot(x=compare_props.index, y="Rand. %error", data=compare_props, palette='magma')
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)
sns.barplot(x=compare_props.index, y="Strat. %error", data=compare_props, palette='magma')

#%%
def drop_duplicates_fromDF(df,subset_col):
   NumberOf_patient_nbr_substract= df.shape[0] - df.drop_duplicates(subset=subset_col, keep="first").shape[0]
   df= df.drop_duplicates(subset=subset_col, keep="first")
   df= df.reset_index(drop=True)
   return df , NumberOf_patient_nbr_substract

drop_duplicates_fromDF(strat_train_set,"patient_nbr") [0]
# df[df['patient_nbr'].duplicated()].shape
# df = df.drop_duplicates(subset = 'patient_nbr', keep = 'first')

# print(len(Maindf.duplicated(subset="patient_nbr", keep='first')))
# NumberOf_patient_nbr_substract= len(Maindf) - len(Maindf.drop_duplicates(subset='patient_nbr', keep="first"))
# Maindf = Maindf.drop_duplicates(subset='patient_nbr', keep="first")

#%%
#this cell is for submitting the Expired and hospice patient from df
discharge_disposition_id_DF = Mapdf[["discharge_disposition_id", "description.1"]]
discharge_disposition_id_DF = discharge_disposition_id_DF[discharge_disposition_id_DF['description.1'].str.contains('Hospice') | discharge_disposition_id_DF['description.1'].str.contains('Expired')]
df = df[~df['discharge_disposition_id'].isin(discharge_disposition_id_DF ['discharge_disposition_id'])]
#%%
display_all(df.info())

#%%
df.describe()

#%%
df.hist(bins=50, figsize=(20,15))

#%%
#Check for NA's in data:
sum_na = df.isnull().sum()
[print(value) for value in sum_na if value > 0] or print("No NA's")

#%%
#Checking percentage of '?' values in each column:
empty_dict ={}
for col in df.columns:
    empty_sum = (df[col] == '?').sum()
    if empty_sum > 0 :
        col_len = len(df[col])
        empty_dict[col] = [empty_sum]
        percentage = np.round(empty_sum/col_len,4)*100
        print(col, empty_dict[col], f'% {percentage:.2f}')

#%%
#Visualization of empty data
sns.set(rc={"figure.figsize":(14, 10)})
colours = ['#34495E', 'seagreen'] 
sns.heatmap(df == '?', cmap=sns.color_palette(colours))

#%%
#Dropping columns with ALOT of missing values:
df = df.drop(['weight', 'medical_specialty', 'payer_code'], axis=1)

#%%
#Checking for columns with just one kind of values,
#can adjust for more values (change the 2 in range function)
unique_dict = {}
col_list = []
for col in df.columns:
    if np.dtype(df[col]) == 'object':
        for i in range(1,2):
            vals = pd.unique(df[col])
            unique_dict[col] = vals
            if len(vals) <= i :
                col_list.append(col)
                print(f'column {col} has {i} unique values', unique_dict[col])
                break

#%%

#Dropping cols with 1 kind of value:
df = df.drop(col_list, axis = 1)
#%%
columns_to_plot = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

# Set up the subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot each column
for i, column in enumerate(columns_to_plot):
    sns.countplot(x=column, data=df, ax=axes[i])
    axes[i].set_title(f'Countplot of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

#%%
#Cleaning IDS_mapping variables:
#These columns have a lot many NA data if different values:
df['admission_type_id'] = df['admission_type_id'].replace([8,6],5)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([18,26],25)
df['admission_source_id'] = df['admission_source_id'].replace([21,20,17,15],9)

# %%
#Grouping diseases by the ID:
diseases = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
ids = ids = [
    (list(range(390, 460)) + [785]),
    (list(range(460, 520)) + [786]),
    (list(range(520, 580)) + [787]),
    np.round(np.arange(250.00, 251.00, 0.01), 2).tolist(),
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
            
# Function to convert values in the 'Value' column based on the ranges
def convert_values(value):
    if value == '?':
        return 'Not Available'
    try:
        numeric_value = float(value)
    except ValueError:
        return diseases[8]  # Skip non-numeric values

    for id, disease in zip(ids, diseases):
        if numeric_value in id:
            return disease  # Replace with the string of your choosing
    return value

#%%

for col in diag_columns:
    df[col] = df[col].apply(convert_values)

#%%

df_melted = pd.melt(df.loc[:,'diag_1':'diag_3'])
plt.figure(figsize=(15,8))

ax = sns.countplot(x='value', hue='variable', data=df_melted)

#%%
listy= []
listy.append(df["diag_1"].value_counts()["Diabetes"])
listy.append(df["diag_2"].value_counts()["Diabetes"])
listy.append(df["diag_3"].value_counts()["Diabetes"])

sum(listy)

#%%
df['A1Cresult'].unique()

##We considered four groups of encounters: (1) no HbA1c
#est performed, (2) HbA1c performed and in normal range,
#(3) HbA1c performed and the result is greater than 8%
#with no change in diabetic medications, and (4) HbA1c
#performed, result is greater than 8%, and diabetic medication
##was changed.

# %%
cond1 = df['A1Cresult'] == 'None'
cond2 = (df['A1Cresult'] == 'Norm')
cond3 = (df['A1Cresult'] == '>7') & (df['change'] == "No")
cond4 = (df['A1Cresult'] == '>7') & (df['change'] == "Ch")
cond5 = (df['A1Cresult'] == '>8') & (df['change'] == "No")
cond6 = (df['A1Cresult'] == '>8') & (df['change'] == "Ch")

# %%
df.loc[cond1, 'A1Cresult'] = 'No HbA1c test performed'
df.loc[cond2, 'A1Cresult'] = 'HbA1c in normal range'
df.loc[cond3, 'A1Cresult'] = 'HbA1c greater than 7%, but no med change'
df.loc[cond4, 'A1Cresult'] = 'HbA1c greater than 7%, with med change'
df.loc[cond5, 'A1Cresult'] = 'HbA1c greater than 8%, but no med change'
df.loc[cond6, 'A1Cresult'] = 'HbA1c greater than 8%, with med change'
# %%
df['A1Cresult'].unique()


