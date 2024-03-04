#%%
"""
this script is used to define the pipeline object and run it
the output of this script is the transformed data
"""
#---------------------------------Imports--------------------------------
import pandas as pd

from DefPipeLineClasses import *
from prepare_data import *
import matplotlib.pyplot as plt

#---------------------------------getting the data--------------------------------
train_set, test_set ,Mapdf= prepare_data_main()

#---------------------------------and preprocessing pipeline--------------------------------

preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)

#---------------------------------get the X train and y train--------------------------------
#in a df mode not a numpy array
df = preprocessing.fit_transform(train_set)
X_train = df.drop("readmitted", axis=1)
y_train = df["readmitted"].copy() 

#---------------------------------get the X test and y test--------------------------------
#in a df mode not a numpy array
df = preprocessing.transform(test_set)
X_test = df.drop("readmitted", axis=1)
y_test = df["readmitted"].copy()

#%%
# --------------------------------- get the X train and y train --------------------------------
# in a numpy array mode not a df
X_train_np = col_processor.fit_transform(X_train)
y_train_np = y_train.to_numpy()
