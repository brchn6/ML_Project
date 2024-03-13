#%%
"""
this script is used to define the pipeline object and run it
the output of this script is the transformed data
"""
#---------------------------------Imports--------------------------------
import pandas as pd
from DefPipeLineClasses import *
from prepare_data import *
from sklearn.compose import make_column_selector as selector
#---------------------------------getting the data--------------------------------
train_set, test_set ,Mapdf= prepare_data_main()
preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)
#----
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
# ---------------------------------Define the GANS preprocessing pipeline--------------------------------
from classes.CopulaGANSyntheticDataGenerator import *
"""
The following function is used to create the preprocessing pipeline with and without the GANS
by spliting the preprocessing pipeline into two parts and adding the GANS in the middle
"""
def build_preprocessing_pipe_withGANS():
    preprocessing1ST = make_pipeline(
        DropDup(dropdup_col),
        DropColumns(columns_to_drop),
        CustomTransformer(bool_functions),
        CopulaGANSyntheticDataGenerator(),
    )
    return preprocessing1ST

def build_preprocessing_pipe_withGANS2ND():
    preprocessing2ND = make_pipeline(
        IDSTransformer(),
        DiseaseConverter(),
        A1CTransformer(),
        CustomTransformer(functions),
    )
    return preprocessing2ND

def Convert_specified_columns_in_the_input_DataFrame_to_object_type(x):
    """
    Convert specified columns in the input DataFrame to object type.

    Args:
        x (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The processed DataFrame with specified columns converted to object type.
    """
    processed = x
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        processed[col] = processed[col].astype(object)
    return processed
prepro_train = Convert_specified_columns_in_the_input_DataFrame_to_object_type(prepro_train)

#---------------------------------Define the main function - name - RunPipe--------------------------------
def RunPipe(train_set, test_set, GANS=False):
    """ doc
    Preprocesses the input data and returns the preprocessed data along with other relevant information.

    Parameters:
    - data: The input data to be preprocessed.
    - GANS: A boolean flag indicating whether to use GANS preprocessing or not. Default is False.

    Returns:
    - If GANS is True, returns the preprocessed data.
    - If GANS is False, returns the preprocessed training and testing data, along with other intermediate variables.
    """
    # ---------------------------------Locigal Q to ask if we want to use GANS or not--------------------------------
    if GANS:
        preprocessingGans = make_pipeline(
            build_preprocessing_pipe_withGANS(), 
            build_preprocessing_pipe_withGANS2ND()
        )
        Pipeline99 = preprocessingGans
    else:
        Pipeline99 = preprocessing

    #---------------------------------preprocess the data--------------------------------
    #---------------------------------make the df var as a DataFrame after the preprocessing pipeline--------------------------------
    """
    train_set --> preprocessing --> df
    """
    #in a df mode not a numpy array
    prepro_train = Pipeline99.fit_transform(train_set)
    prepro_train = Convert_specified_columns_in_the_input_DataFrame_to_object_type(prepro_train)

    #---------------------------------get the X train and y train--------------------------------
    X_train = prepro_train.drop("readmitted", axis=1)
    y_train = prepro_train["readmitted"].copy() 

    #---------------------------------get the X test and y test--------------------------------
    #in a df mode not a numpy array
    prepro_test = preprocessing.transform(test_set)
    X_test = prepro_test.drop("readmitted", axis=1)
    y_test = prepro_test["readmitted"].copy()

    # --------------------------------- get the X train and y train --------------------------------
    X_train = Convert_specified_columns_in_the_input_DataFrame_to_object_type(X_train)
    X_test = Convert_specified_columns_in_the_input_DataFrame_to_object_type(X_test)
    
    # --------------------------------- Get X_train_np and y_train_np --------------------------------
    """
    In order to implament the data into ML model, we need to convert the data into
    data that can goes into onehotencoder and standard scaler for the categorical and
    standard scaler for the numerical data
    to do that we will use the col_processor pipeline which we wrote in the DefPipeLineClasses.py
    on the X_train and y_train to get the X_train_np and y_train_np
    """
    # in a numpy array mode not a df
    X_train_np = col_processor.fit_transform(X_train)
    X_test_np = col_processor.transform(X_test)
    
    return X_train, y_train, X_test, y_test, prepro_train, prepro_test ,X_train_np, X_test_np ,Pipeline99
        
    # if we wish to see the data in a df mode we can use the following code
def show_data():
    print("X_train_np:")
    display(pd.DataFrame(X_train_np.toarray()))
    print("X_test_np:")
    display(pd.DataFrame(X_test_np.toarray()))
# show_data()

# ---------------------------------Run the main function--------------------------------
X_train, y_train, X_test, y_test, prepro_train, prepro_test ,X_train_np, X_test_np ,Pipeline99 = RunPipe(train_set, test_set, GANS=False)