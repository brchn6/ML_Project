#%%
"""
Main file for the BarModels directory
"""

#---------------------------- Imports -------------------------------
import numpy as np
import os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# ---------------------------- data incoming -------------------------------
X_train_np = np.load("X_train_np.npy", allow_pickle=True).item()
y_train = np.load("./y_train.npy")

# ---------------------------- Rendom_forest -------------------------------
#create a Rendom_forest_classification_BC object
from Rendom_forest import Rendom_forest_classification_BC
rf = Rendom_forest_classification_BC(X_train_np, y_train, X_train_np, y_train)
