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
from classes.CopulaGenerator import *
from classes.ConditionalTransformer import *
#---------------------------------getting the data--------------------------------
train_set = train_set
test_set = test_set 
Mapdf = Mapdf

#%%
#---------------------------------Define the GANS preprocessing pipeline--------------------------------
"""
If no GANS is needed, the set condition atrribute to False inside ConditionalTransformer.
If GANS is needed, the set condition atrribute to True inside ConditionalTransformer
&
set the transformer attributes to the GANS object
"""
copula_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                               majority_class_label=1, 
                                               gans = 'ctgan',
                                               boolean_columns=['change', 'diabetesMed'], 
                                               epochs=50)

preprocessing = make_pipeline(
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
    ConditionalTransformer(condition=False, transformer=copula_gans),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)

prepro_train = preprocessing.fit_transform(train_set)
#%%
#----
#Converts specific columns to object type:

def convertIdsColumn(x):
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

"""
this pipeline is used to
make the data ready to be used in the ML model
output: X_train_np, y_train_np (np array)
"""
col_processor = make_column_transformer(
    (num_transformer, selector(dtype_include="number")),
    (bool_transformer, selector(dtype_include="bool")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=3
)
#%%
#---------------------------------preprocess the data--------------------------------
#---------------------------------make the df var as a DataFrame after the preprocessing pipeline--------------------------------
"""
train_set --> preprocessing --> df
"""
#in a df mode not a numpy array
prepro_train = preprocessing.fit_transform(train_set)
prepro_train = convertIdsColumn(prepro_train)
#---------------------------------get the X train and y train--------------------------------
X_train = prepro_train.drop("readmitted", axis=1)
y_train = prepro_train["readmitted"].copy() 
#---------------------------------get the X test and y test--------------------------------
#in a df mode not a numpy array
prepro_test = preprocessing.transform(test_set)
X_test = prepro_test.drop("readmitted", axis=1)
y_test = prepro_test["readmitted"].copy()
# --------------------------------- get the X train and y train --------------------------------
X_train = convertIdsColumn(X_train)
X_test = convertIdsColumn(X_test)

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

# if we wish to see the data in a df mode we can use the following code
def show_data():
    print("X_train_np:")
    display(pd.DataFrame(X_train_np.toarray()))
    print("X_test_np:")
    display(pd.DataFrame(X_test_np.toarray()))
# show_data()


