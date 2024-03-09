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
from sklearn.compose import make_column_selector as selector
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
#---------------------------------Define column processor--------------------------------
"""
this pipeline is used to
make the data ready to be used in the ML model
output: X_train_np, y_train_np (np array)
"""
col_processor = make_column_transformer(
    (num_transformer, selector(dtype_include="number")),
    (bool_transformer, selector(dtype_include="bool")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=3,
)
#---------------------------------make the df var as a dataframe afrer the preprocessing pipeline--------------------------------
"""
train_set --> preprocessing --> df
"""
#in a df mode not a numpy array
prepro_train = preprocessing.fit_transform(train_set)
"""
We need to change something in the initial transformations, 
leave it like this for now.
now its hardcode but we will change it to be dynamic in the future
"""
def randomfun(x):
    processed = x
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        processed[col] = processed[col].astype(object)
    return processed
prepro_train = randomfun(prepro_train)

#---------------------------------get the X train and y train--------------------------------
X_train = prepro_train.drop("readmitted", axis=1)
y_train = prepro_train["readmitted"].copy() 

#---------------------------------get the X test and y test--------------------------------
#in a df mode not a numpy array
prepro_test = preprocessing.transform(test_set)
X_test = prepro_test.drop("readmitted", axis=1)
y_test = prepro_test["readmitted"].copy()

X_train = randomfun(X_train)
X_test = randomfun(X_test)

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


"""
The following function is used to create the preprocessing pipeline with and without the GANS
by spliting the preprocessing pipeline into two parts and adding the GANS in the middle
"""
def build_preprocessing_pipe_withGANS():
    preprocessing1ST = make_pipeline(
        DropDup(dropdup_col),
        DropColumns(columns_to_drop),
        CustomTransformer(bool_functions),
        
    )
    return preprocessing1ST

def build_preprocessing_pipe_withGANS2ND():
    preprocessing2ND = make_pipeline(
        IDSTransformer(),
        DiseaseConverter(),
        A1CTransformer(),
        CustomTransformer(functions),
    )

def ask_use_gans():
    response = input("Do you want to use GANS? (True/False): ")
    return response.strip().lower() == 'true'

def CombinePipeLine(data, use_gans=None):
    # If use_gans is None, ask the user; otherwise, use the provided argument
    if use_gans is None:
        use_gans = ask_use_gans()

    if use_gans:
        # Assuming build_preprocessing_pipe_withGANS and build_preprocessing_pipe_withGANS2ND are defined
        preprocessingGans = make_pipeline(
            build_preprocessing_pipe_withGANS(), 
            build_preprocessing_pipe_withGANS2ND()
        )
        return preprocessingGans.fit_transform(data)
    else:
        # Assuming all other transformers are defined and 'dropdup_col', 'columns_to_drop', 'functions' are available
        preprocessing = make_pipeline(
            DropDup(dropdup_col),
            DropColumns(columns_to_drop),
            IDSTransformer(),
            DiseaseConverter(),
            A1CTransformer(),
            CustomTransformer(functions),
        )
        return preprocessing.fit_transform(data)
    
# ask_use_gans()
