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
#%%

# Set numerical columns
num_cols = ['num_medications', 'num_lab_procedures']

# Set categorical columns
cat_cols = [col for col in diabetes_test.columns if col not in num_cols]

# Define preprocessing steps for numerical and categorical columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Create a pipeline with preprocessing and mapping
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)])

# Fit and transform the data
diabetes_prepared = pipeline.fit_transform(diabetes_test)

#see the transformed data
pd.DataFrame(diabetes_prepared.toarray())

# Get the feature names after one-hot encoding
feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)

# Create a dictionary mapping feature names to their indices
feature_indices = {feature_names[i]: i for i in range(len(feature_names))}

#start count from the length of the numerical columns
for k, v in feature_indices.items():   
    feature_indices[k] = v + len(num_cols)

