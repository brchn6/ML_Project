#%%
"""
this script is used to define the pipeline object and run it
the output of this script is the transformed data
"""
#---------------------------------Imports--------------------------------
import pandas as pd
from DefPipeLineClasses import *
from prepare_data import *
from sklearn.compose import make_column_selector as selector
from classes.CopulaGANSyntheticDataGenerator import *
from classes.ConditionalTransformer import *
#---------------------------------getting the data--------------------------------
train_set, test_set ,Mapdf= prepare_data_main()

cupula_gans = CopulaGANSyntheticDataGenerator()

preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
    CopulaGANSyntheticDataGenerator(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)
#----
"""
this pipeline is used to
make the data ready to be used in the ML model
output: X_train_np, y_train_np (np array)
"""
col_processor = make_column_transformer(
    (num_transformer, selector(dtype_include="number")),
    (bool_transformer, selector(dtype_include="bool")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=3,
)