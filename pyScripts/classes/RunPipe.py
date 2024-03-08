#%%
"""
this script is used to define the pipeline object and run it
the output of this script is the transformed data
"""
#---------------------------------Imports--------------------------------
import pandas as pd
from sklearn.pipeline import make_pipeline
from pyScripts.__main__ import *

#%%
from .DefPreprocessing_PipeLineClasses import * #this is the file that contains the classes for the preprocessor pipeline
from .DefTreanformer_PipeLineClasses import * #this is the file that contains the classes for the col_processor pipeline
from .disease_ids_conds import * #this is the file that contains the functions for the CustomTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
#---------------------------------Define the Hardcoded variables--------------------------------
dropdup_col = "patient_nbr"
columns_to_drop = ['payer_code', 'encounter_id', 'weight', 'patient_nbr', 'medical_specialty'] + ['acetohexamide', 'troglitazone', 'examide', 'citoglipton', 'metformin-rosiglitazone','max_glu_serum']
#---------------------------------and preprocessing pipeline--------------------------------
"""
create an instance of the make_pipeline class
name preprocessing
this instance is used to apply the pre-processing functions in the __main__.py file
"""
preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)
#---------------------------------Define column processor--------------------------------
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
