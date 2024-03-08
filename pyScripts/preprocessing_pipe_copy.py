#%%
import pandas as pd
import numpy as np
import os
import matplotlib as plt
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_transformer
from imblearn.pipeline import Pipeline as impipe
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_impipe
from sklearn.compose import make_column_selector as selector

from PipeLineObject_V2 import *
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())

# Define preprocessing steps for numerical and categorical columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

bool_transformer = Pipeline(steps=[
    ('booltransform', BooleanConverter())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',drop='if_binary'))])

# Create a column transformer to apply different preprocessing steps to different column types
col_processor_s = make_column_transformer(
    (num_transformer, selector(dtype_include="number")),
    (bool_transformer, selector(dtype_include="bool")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=3,
)

# Define a pipeline to apply custom preprocessing steps to the dataset
preprocessing_s = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions)
)

# Apply preprocessing steps to the training dataset 
processed = preprocessing_s.fit_transform(train_set)

#We need to change something in the initial transformations, leave it like this for now.
for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
    processed[col] = processed[col].astype(object)


