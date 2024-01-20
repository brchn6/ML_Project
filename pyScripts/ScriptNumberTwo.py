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
    sys.path.append(os.path.join(GETCWD, "/pyScripts"))


pd.set_option("display.max_row", 100) #add a option of pd
pd.set_option("display.max_columns", 100) #add a option of pd

from ScriptNumberOne import pullDF
train_set = pullDF()
from ScriptNumberOne import train_set
