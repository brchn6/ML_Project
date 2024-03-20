#%%
"""
Main file for the BarModels directory
"""
%load_ext autoreload
%autoreload 2

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
#build the model
classifier_fit = rf.build_RandomForestClassifier()
#predict the model
predictions = rf.predict_RandomForestClassifierTrainData(classifier_fit)
#check the accuracy
#%%
accuracy = rf.accuracy_score(predictions)
print(accuracy)
predictions = rf.predict_RandomForestClassifierTestData(classifier_fit)
print(predictions)

#%%
#cv



grid= rf.gridSearchCV_RandomForestClassifier()

print(grid.best_params_)
