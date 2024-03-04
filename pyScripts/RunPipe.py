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

"""
In order to implament the data into ML model, we need to convert the data into
data that can goes into onehotencoder and standard scaler for the categorical and
standard scaler for the numerical data
to do that we will use the col_processor pipeline which we wrote in the DefPipeLineClasses.py
on the X_train and y_train to get the X_train_np and y_train_np
"""
# --------------------------------- get the X train and y train --------------------------------
# in a numpy array mode not a df
X_train_np = col_processor.fit_transform(X_train)
X_test_np = col_processor.transform(X_test)

# if we wish to see the data in a df mode we can use the following code
def show_data():
    print("X_train_np:")
    display(pd.DataFrame(X_train_np.toarray()))
    print("X_test_np:")
    display(pd.DataFrame(X_test_np.toarray()))


def main():
    show_data()

#to call the main function remove the comment from the next line
# main()
