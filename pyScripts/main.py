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
from deadendscript.CopulaGenerator import CopulaGANSyntheticDataGenerator
#%%
# Specify columns to transform

#train_set = build_preprocessing_pipe_withGANS().fit_transform(train_set)

boolean_columns = ['diabetesMed', 'change']
label_column = 'readmitted'

generator = CopulaGANSyntheticDataGenerator(train_set, label_column, 0, 1, boolean_columns=boolean_columns, enforce_min_max_values=True)
synthetic_samples = generator.fit_and_generate()
quality_score = generator.evaluate_quality(synthetic_samples)
diagnostic_score = generator.run_diagnostic(synthetic_samples)
balanced_train_set = generator.generate_balanced_df(synthetic_samples, train_set)
generator.export_balanced_df(balanced_train_set)
#%%
"""
This part we are implamenting the GANS pipeline
"""
ask_use_gans()
pipe= CombinePipeLine(train_set,use_gans=None)


#%%
#create an instance of the claafication class
CL_instance = Rendom_forest_classification_BC(X_train_np, y_train, X_test_np, y_test)
#make the regressor
classifier = CL_instance.build_RandomForestClassifier()
# predictions
CL_instance.predict_RandomForestClassifierTrainData(classifier)
CL_instance
