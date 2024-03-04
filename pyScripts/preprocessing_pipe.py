#%%
"""
this script is used to define the pipeline object and run it
the output of this script is the transformed data
"""
import sys
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from PipeLineClasses import *
import matplotlib.pyplot as plt

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

# Fit and transform the training set
processed = preprocessing.fit_transform(train_set)