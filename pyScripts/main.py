#%%
""""
This is the fucking main file, all file should be called from here
"""
# ---------------------------- Imports -------------------------------
from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
from AddRootDirectoriesToSysPath import AddRootDirectoriesToSys #importing the funcrion htat add all path in the dir to sys
AddRootDirectoriesToSys() #implament this function
import pandas as pd


#%%
# ------------------------------Run Rendom_forest.py script ------------------------------d
#importing the classes and var from BarModels.Rendom_forest
from BarModels.Rendom_forest import *

#%%
#creat an instence of the calss i build
RF_instance  = Rendom_forest_regression_BC(X_train_np,y_train,X_test_np,y_test)
#make the regressor
regressor = RF_instance.build_RandomForestRegressor()
# Predictions
predictions = RF_instance.predict_RandomForestRegressor(regressor)

#%%
errors = abs(predictions - y_test)
errors 
#plot
import matplotlib.pyplot as plt

plt.scatter(y_test, errors)
plt.xlabel('Original Data')
plt.ylabel('Error')
plt.title('Error vs Original Data')
plt.show()

#%%
X_train = pd.read_csv('../data/copula_train_set.csv')
y_train = X_train['readmitted']
X_train = X_train.drop('readmitted', axis=1)

X_train = col_processor.fit_transform(X_train)
X_train



#%%
#create an instance of the claafication class
CL_instance = Rendom_forest_classification_BC(X_train_np, y_train, X_test_np, y_test)
#make the regressor
classifier = CL_instance.build_RandomForestClassifier()
# predictions
CL_instance.predict_RandomForestClassifierTrainData(classifier)
CL_instance
