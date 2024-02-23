import pandas as pd
import numpy as np
import os
import matplotlib as plt
import sys
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())

#from sklearn.pipeline import Pipeline


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from sklearn.compose import make_column_transformer
from PipeLineObject_V2 import *


smote = SMOTENC(random_state=42,categorical_features=cat_cols)


# Define preprocessing steps for numerical and categorical columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

#cat_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='most_frequent')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',drop='if_binary'))])

col_processor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)],remainder='passthrough')

data_processor = Pipeline([
    ('dropdup', DropDup(dropdup_col)),
    ('dropcols', DropColumns(columns_to_drop)),
    ('idstransform', IDSTransformer()),
    ('convertdisease', DiseaseConverter()),
    ('a1ctransform',A1CTransformer()),
    ('customcols', CustomTransformer(functions)),
    ('label_fetch',LabelFetcher())])

preprocessing = Pipeline([
    ('dropdup', DropDup(dropdup_col)),
    ('dropcols', DropColumns(columns_to_drop)),
    ('idstransform', IDSTransformer()),
    ('convertdisease', DiseaseConverter()),
    ('a1ctransform',A1CTransformer()),
    ('customcols', CustomTransformer(functions)),
    ('col_spec',col_processor)])

from imblearn.pipeline import Pipeline as impipe
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_impipe


from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer

col_processor_s = make_column_transformer(
    (num_transformer, selector(dtype_include="number")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=2,
)

preprocessing_s = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions))

processed = preprocessing_s.fit_transform(train_set)

#X_train = processed.copy()
#y_train = processed['readmitted'].astype('category')
#X_train = X_train.drop(columns='readmitted')

#pp = make_impipe(
#        smote,
#        col_processor_s)
#xx = pp.fit_transform(X_train,y_train)


################################################
#ohe_feature_names = preprocessing.named_steps['col_spec']\
#                              .named_transformers_['cat']\
#                              .named_steps['onehot']\
#                              .get_feature_names_out(input_features=cat_cols)\
#
#one_hot_cols = list(ohe_feature_names)
#all_feature_names = num_cols + list(ohe_feature_names)
#train_set_processed_df.columns = all_feature_names
