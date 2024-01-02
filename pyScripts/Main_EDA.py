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


#%%
df = Maindf
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

# %%
#Visualization of empty data
sns.set(rc={"figure.figsize":(14, 10)})
colours = ['#34495E', 'seagreen'] 
sns.heatmap(Maindf == '?', cmap=sns.color_palette(colours))
#%%
#Checking for columns with just one kind of values,
#can adjust for more values (change the 2 in range function)
unique_dict = {}
for col in Maindf.columns:
    for i in range(1,2):
        vals = pd.unique(Maindf[col])
        unique_dict[col] = vals
        if len(vals) <= i :
            print(f'column {col} has {i} unique values', unique_dict[col])
            break
#%%