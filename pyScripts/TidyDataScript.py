#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

#path to data file
GETCWD = os.getcwd()
PathToData = os.path.join(GETCWD + "\\diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

#assing df
Maindf = pd.read_csv(PathToData)

sns.set_style("darkgrid")
plt.style.use("dark_background")
#%%
