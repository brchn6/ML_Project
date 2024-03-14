#%%
""""
This is the fucking main file, all file should be called from here
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------- Imports -------------------------------
from RunPipe import * #get all the data from the RunPiple.py file 
#data such as x train, yratin, xtset ect
from AddRootDirectoriesToSysPath import AddRootDirectoriesToSys #importing the funcrion htat add all path in the dir to sys
AddRootDirectoriesToSys() #implament this function

#%%
from classes.CopulaGenerator import CopulaGANSyntheticDataGenerator

#%%
# Specify columns to transform

#train_set = build_preprocessing_pipe_withGANS().fit_transform(train_set)

boolean_columns = ['diabetesMed', 'change']
label_column = 'readmitted'

"""
generator = CopulaGANSyntheticDataGenerator(train_set, label_column, 0, 1, boolean_columns=boolean_columns, enforce_min_max_values=True)
synthetic_samples = generator.fit_and_generate()
quality_score = generator.evaluate_quality(synthetic_samples)
diagnostic_score = generator.run_diagnostic(synthetic_samples)
balanced_train_set = generator.generate_balanced_df(synthetic_samples, train_set)
generator.export_balanced_df(balanced_train_set)
"""
train_data = preprocessing.fit_transform(train_set)
generator = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0, majority_class_label=1, boolean_columns=boolean_columns)
aa = generator.fit_transform(train_data, export = True)
#%%
"""
This part we are implamenting the GANS pipeline
"""
ask_use_gans()
pipe= CombinePipeLine(train_set,use_gans=None)

