#%%
""""
This is the fucking main file, all file should be called from here
"""
# ---------------------------- Imports -------------------------------
from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
from AddRootDirectoriesToSysPath import AddRootDirectoriesToSys #importing the funcrion htat add all path in the dir to sys
AddRootDirectoriesToSys() #implament this function

#%%
# ------------------------------Run Rendom_forest.py script ------------------------------d
#importing the classes and var from BarModels.Rendom_forest
from BarModels.Rendom_forest import *
#creat an instence of the calss i build
RF_instance  = Rendom_forest_regression_BC(X_train_np,y_train,X_test_np,y_test)
#make the regressor
regressor = RF_instance.build_RandomForestRegressor()

# #%%
# # ------------------------------Run Rendom_forest.py script ------------------------------d
# from BarModels.Rendom_forest import *
# regressor  = build_RandomForestRegressor(X_train_np,y_train)
# predictions = predict_RandomForestRegressor(regressor,X_test_np)


# #%%
# errors = abs(predictions - y_test)
# errors 

