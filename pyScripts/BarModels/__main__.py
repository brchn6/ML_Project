#%%
"""
Main file for the BarModels directory
"""
import warnings

warnings.filterwarnings('ignore')
%reload_ext autoreload
%autoreload 2

#---------------------------- Imports -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# ---------------------------- data incoming -------------------------------
#load the train ds
X_train_np = np.load("X_train_np.npy", allow_pickle=True).item()
y_train = np.load("./y_train.npy")
#load the test ds
X_test_np = np.load("X_test_np.npy", allow_pickle=True).item()
y_test = np.load("./y_test.npy")
#%%
# ---------------------------- Rendom_forest -------------------------------
#importing the Rendom_forest_classification_BC_defultParams class
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_defultParams
#create a Rendom_forest_classification_BC object
rf = Rendom_forest_classification_BC_defultParams(X_train_np, y_train, X_train_np, y_train)
#build the model
classifier_fit = rf.build_RandomForestClassifier()
#predict the model
predictions = rf.predict_RandomForestClassifierTrainData(classifier_fit)
#check the accuracy
accuracy = rf.accuracy_score(predictions)
print(accuracy)
print(predictions)
#%%
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_useingGridSearchCV
rf = Rendom_forest_classification_BC_useingGridSearchCV(X_train_np, y_train, X_train_np, y_train)
best_rf_classifier= rf.gridSearchCV_RandomForestClassifier()
#%%
# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 
rfc = RandomForestClassifier(random_state=42)
# fit the model
rfc.fit(X_train_np, y_train)
# Predict the Test set results

y_pred = rfc.predict(X_train_np)
# Check accuracy score 
from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))