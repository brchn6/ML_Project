"""
This script was used to test the feature importance on the copulaGANS generated balanced train set.
The version of the train set used was the one with 4 numeric features 'num_lab_procedures', 'num_medications', 'number_emergency' and number_outpatient'.
"""
#%%
#Import the necessary libraries:
import os
#from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
#get name of current working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.getcwd())
#from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from classes.ConditionalTransformer import *
from DefPipeLineClasses import *
from classes.CopulaGenerator import *
from DefPipeLineClasses import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector as selector
#%%
#Call the train_set from RunPipe.py:
"""
train_set = train_set

#%%
%reload_ext autoreload
%autoreload 2
#Redfine the functions from deadendscript/disease_ids_conds.py:
functions = {
    'time_in_hospital': timeInHosp,
    'num_procedures': numProcedures,
    'number_inpatient': inPatiant,
    'number_diagnoses': numDiag,
    'race' : raceChange,
    'age': ageFunc
}

#Define copula_gans object:
copula_gans = None

#Define the preprocessing pipeline:  
preprocessing = make_pipeline(
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
    ConditionalTransformer(condition=False, transformer=copula_gans),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)

#fit and transform the train_set:
df_transformed_test = preprocessing.fit_transform(test_set)
df_name =  'copula_gans_4_numeric_TestSet'

#%%
#creat the path to save the transformed Test set:
path = os.path.join(os.getcwd(),'../', 'data', df_name + '.csv')
print(path)
#save the transformed Test set:
df_transformed_test.to_csv(path, index=False)
"""