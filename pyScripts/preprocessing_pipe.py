#%%
import pandas as pd
import numpy as np
import os
import sys
from AddRootDirectoriesToSysPath import add_directories_to_sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from PipeLineObject import *
import matplotlib.pyplot as plt

# Add root directories to sys path
add_directories_to_sys(os.getcwd())

# Define SMOTENC object for oversampling
smote = SMOTENC(random_state=42, categorical_features=cat_cols)

# Define preprocessing steps for numerical and categorical columns
num_transformer = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

cat_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', drop='if_binary')
)

# Define column processor
col_processor = make_column_transformer(
    (num_transformer, make_column_selector(dtype_include="number")),
    (cat_transformer, make_column_selector(dtype_include="object")),
    remainder='passthrough'
)

# Define data processor pipeline
data_processor = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
    LabelFetcher()
)

# Define preprocessing pipeline
preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
    col_processor
)

# Define simplified column processor
col_processor_s = make_column_transformer(
    (num_transformer, make_column_selector(dtype_include="number")),
    (cat_transformer, make_column_selector(dtype_include="object")),
    n_jobs=2
)

# Define simplified preprocessing pipeline
preprocessing_s = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions)
)

# Fit and transform the training set
processed = preprocessing.fit_transform(train_set)