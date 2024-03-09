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

#%%
from deadendscript.name import CopulaGANSyntheticDataGenerator
#%%
# Specify columns to transform

train_set = build_preprocessing_pipe_withGANS().fit_transform(train_set)

boolean_columns = ['diabetesMed', 'change']
label_column = 'readmitted'

generator = CopulaGANSyntheticDataGenerator(train_set, label_column, 0, 1, boolean_columns=boolean_columns, enforce_min_max_values=True)
synthetic_samples = generator.fit_and_generate()
quality_score = generator.evaluate_quality(synthetic_samples)
diagnostic_score = generator.run_diagnostic(synthetic_samples)
balanced_train_set = generator.generate_balanced_df(synthetic_samples, train_set)
generator.export_balanced_df(balanced_train_set)
