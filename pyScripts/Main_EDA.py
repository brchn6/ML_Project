#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import sys 
# Define the path you want to add
path_to_add = "/c/Users/barc/Dropbox (Weizmann Institute)/MSc_Weizmann/FGS_ML/ML_Project/pyScripts/"

# Add the path to sys.path if it's not already there
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
from SeeTheData import SeeTheData

#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "/.." + "\\diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv" )

#assing df
Maindf = pd.read_csv(PathToData)

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
df["discharge_disposition_id"][0:100]
#%%
# Simulate data from a bivariate Gaussian
x = dfDiag["diag_1"]
y = dfDiag["diag_3"]

# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
# sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)