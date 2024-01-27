import pandas as pd
import numpy as np
import os
import matplotlib as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from test import *

initial_pipeline = Pipeline([
        ('dropdup', DropDup(dropdup_col)),
        ('dropcols', DropColumns(columns_to_drop)),
        ('convertdisease', DiseaseConverter()),
        ('idstransform', IDSTransformer()),
        ('a1ctransform',A1CTransformer()),
        ('customcols', CustomTransformer(functions)),
    ])

diabetes_test = initial_pipeline.fit_transform(train_set_mod)



num_cols = ['num_medications', 'num_lab_procedures']
df_num = diabetes_test.drop(num_cols, axis = 1)

cat_cols = list(df_num)

full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols),
    ])

diabetes_prepared = full_pipeline.fit_transform(diabetes_test)

