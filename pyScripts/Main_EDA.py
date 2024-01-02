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
#%%

# Define the path you want to add
path_to_add = "/c/Users/barc/Dropbox (Weizmann Institute)/MSc_Weizmann/FGS_ML/ML_Project/pyScripts/"

# Add the path to sys.path if it's not already there
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
from SeeTheData import SeeTheData

#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "/.." + "\\diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv" )
PathToMap = os.path.join(GETCWD + "/.." + "\\diabetes+130-us+hospitals+for+years+1999-2008/IDS_mapping.csv")

#assing df
Maindf = pd.read_csv(PathToData)
mapdf = pd.read_csv(PathToMap)

sns.set_style("darkgrid")
plt.style.use("dark_background")

# [func for func in dir(SeeTheData) if callable(getattr(SeeTheData, func))]

#%%
a= SeeTheData(Maindf)
a.Subsetting()
# a.Display()
# a.CountPlotOfObjectColumns()
a.HistPlotOfNumericColumns()
#%%
# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
# sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)


#%%
df = Maindf

#%%
#Removing non-diabetes diagnosis should be before starting EDA
test_df = df[df['diag_1'].str.contains('250') | df['diag_2'].str.contains('250') | df['diag_3'].str.contains('250')]

# %%
#display all data:
def display_all(data):
    with pd.option_context("display.max_row", 100, "display.max_columns", 100):
        display(data)
display_all(mapdf)

#%%
Maindf.info()

#%%
Maindf.describe()

#%%
Maindf.hist(bins=50, figsize=(20,15))

#%%

sns.countplot(Maindf)

#%%
#Check for NA's in data:
sum_na = Maindf.isnull().sum()
[print(value) for value in sum_na if value > 0] or print("No NA's")

#%%
#Checking percentage of '?' values in each column:
empty_dict ={}
for col in Maindf.columns:
    empty_sum = (Maindf[col] == '?').sum()
    if empty_sum > 0 :
        col_len = len(Maindf[col])
        empty_dict[col] = [empty_sum]
        print(col, empty_dict[col], f'% {np.round(empty_sum/col_len,2)*100}')

#%%
#Visualization of empty data
sns.set(rc={"figure.figsize":(14, 10)})
colours = ['#34495E', 'seagreen'] 
sns.heatmap(Maindf == '?', cmap=sns.color_palette(colours))

#%%
#Dropping columns with ALOT of missing values:
Maindf = hospDf.drop(['weight', 'medical_specialty', 'payer_code'], axis=1)

#%%
#Checking for columns with just one kind of values,
#can adjust for more values (change the 2 in range function)
unique_dict = {}
for col in Maindf.columns:
    if np.dtype(Maindf[col]) == 'object':
        for i in range(1,2):
            vals = pd.unique(Maindf[col])
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
    sns.countplot(x=column, data=Maindf, ax=axes[i])
    axes[i].set_title(f'Countplot of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

#%%
#Cleaning IDS_mapping variables:
#These solumns have alot many NA data with different values:
df['admission_type_id'] = df['admission_type_id'].replace([8,6],5)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([11,18,26],25)
df['discharge_disposition_id'] = df['discharge_disposition_id'].replace([21,20,17,15],9)
