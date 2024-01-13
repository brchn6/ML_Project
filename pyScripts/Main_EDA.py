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
pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd
#%%
# Define the path you want to add
path_to_add = "/c/Users/barc/Dropbox (Weizmann Institute)/MSc_Weizmann/FGS_ML/ML_Project/pyScripts/"

# Add the path to sys.path if it's not already there
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
from SeeTheData import SeeTheData

#%% importing data
#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "/.." + "\\diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv" )
PathToMap = os.path.join(GETCWD + "/.." + "\\diabetes+130-us+hospitals+for+years+1999-2008/IDS_mapping.csv")

#assing df
Maindf = pd.read_csv(PathToData)
Mapdf = pd.read_csv(PathToMap)

#sns + plt option and settings
sns.set_style("darkgrid")
plt.style.use("dark_background")

#%% SeeTheData script OOP will be use in the future
# a= SeeTheData(Maindf)
# a.Subsetting()
# a.Display()
# a.CountPlotOfObjectColumns()
# a.HistPlotOfNumericColumns()
#%%
df = Maindf
#Removing non-diabetes diagnosis should be before starting EDA
Subset_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]
df = Subset_df
# %%
#display all data:
def display_all(data):
    with pd.option_context("display.max_row", 100, "display.max_columns", 100):
        display(data)
#%%
np.random.seed(42)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(df, 0.2)
print(len(train_set))
print(len(test_set))
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
for col in df.columns:
    if np.dtype(df[col]) == 'object':
        for i in range(1,2):
            vals = pd.unique(df[col])
            unique_dict[col] = vals
            if len(vals) <= i :
                print(f'column {col} has {i} unique values', unique_dict[col])
                break
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
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([11,18,26],25)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([21,20,17,15],9)

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

df_melted = pd.melt(df1.loc[:,'diag_1':'diag_3'])
plt.figure(figsize=(15,8))

ax = sns.countplot(x='value', hue='variable', data=df_melted)

#%%
listy= []
listy.append(df1["diag_1"].value_counts()["Diabetes"])
listy.append(df1["diag_2"].value_counts()["Diabetes"])
listy.append(df1["diag_3"].value_counts()["Diabetes"])

sum(listy)

#%%
df['readmitted'].unique()

#%%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# %%
fig, ax = plt.subplots(1,3,figsize = (10,8))

ax[0].hist(df['insulin'])
ax[1].hist(train_set['insulin'])
ax[2].hist(test_set['insulin'])

# %%
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
#%%
df