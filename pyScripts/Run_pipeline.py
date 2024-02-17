#%%
# Description: This script is used to test the pipeline created in the pipelineObject.py

import pandas as pd
import numpy as np
import os
import matplotlib as plt
import sys
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from PipeLineObject import *

# Create the first pipeline
preprocess_pipeline = Pipeline([
    ('dropdup', DropDup(dropdup_col)),
    ('dropcols', DropColumns(columns_to_drop)),
    ('idstransform', IDSTransformer()),
    ('labe', LabelFetcher()),
],memory = None)

# Fit and transform the pipeline on the training data
diabetes_labels, train_set_mod = preprocess_pipeline.fit_transform(train_set)

initial_pipeline = Pipeline([
        ('convertdisease', DiseaseConverter()),
        ('a1ctransform',A1CTransformer()),
        ('customcols', CustomTransformer(functions)),
    ])

diabetes_test = initial_pipeline.fit_transform(train_set_mod)

#seeting columns that are numerical and not categorical
num_cols = ['num_medications', 'num_lab_procedures']
#dropping the numerical columns from the DataFrame
df_num = diabetes_test.drop(num_cols, axis=1)

#setting the categorical columns
cat_cols = list(df_num)

#running the pipeline
full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols),
    ])

#calling the fit_transform method to transform the diabetes_test dataframe
diabetes_prepared = full_pipeline.fit_transform(diabetes_test)
