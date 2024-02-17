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

preprocess_pipeline = Pipeline([
    ('dropdup', DropDup(dropdup_col)),
    ('dropcols', DropColumns(columns_to_drop)),
    ('labe', LabelFetcher()),
])

# Assuming train_set is your training data
b, a = preprocess_pipeline.fit_transform(train_set)


#%%

initial_pipeline = Pipeline([
        ('dropdup', DropDup(dropdup_col)),
        ('dropcols', DropColumns(columns_to_drop)),
        ('convertdisease', DiseaseConverter()),
        ('idstransform', IDSTransformer()),
        ('a1ctransform',A1CTransformer()),
        ('customcols', CustomTransformer(functions)),
    ])

diabetes_test = initial_pipeline.fit_transform(train_set_mod)

#see the diabetes_test as display not as df
#pd.DataFrame(diabetes_test)


#seeting colums that are numerical and not categorical
num_cols = ['num_medications', 'num_lab_procedures']
#droping the numerical columns from the dataframe
df_num = diabetes_test.drop(num_cols, axis = 1)

#setting the categorical columns
cat_cols = list(df_num)

#running the pipeline
full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols),
    ])

#calling the fit_transform method to transform the diabetes_test dataframe
diabetes_prepared = full_pipeline.fit_transform(diabetes_test)


#%%
#see the diabetes_prepared array as display not as <25468x178 sparse matrix of type '<class 'numpy.float64'>' with 993252 stored elements in Compressed Sparse Row format>
pd.DataFrame(diabetes_prepared.toarray())

#%%
diabetes_labels.shape
diabetes_prepared.shape