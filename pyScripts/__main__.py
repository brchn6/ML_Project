#%%
"""
This is the main file for the pyScripts package. It is the entry point for the package.
"""

#------------------------------Imports---------------------------------
from .classes.GetTheData import PathToData, PathToMap
from .classes.PreProssesingFunction import PreProcessingFunctions
import pandas as pd
from .classes.RunPipe import preprocessing, col_processor     

#apply the pre-processing functions
def main():
    #------------------------------path to DataFrames----------------
    data = pd.read_csv(PathToData)
    map = pd.read_csv(PathToMap)
    #------------------------------Pre-processing----------------
    pre = PreProcessingFunctions()
    data = pre.remove_unwanted_columns_and_rows(data)
    train_set, test_set = pre.split_data(data, "readmitted")
    return data, map , train_set, test_set

if __name__ == "__main__":
    data, Mapdf , train_set, test_set = main()

#%%
###########################
#---------------------------------make the df var as a dataframe afrer the preprocessing pipeline--------------------------------
"""
train_set --> preprocessing --> df
"""
#in a df mode not a numpy array
#%%
# prepro_train = preprocessing.fit_transform(train_set)

# """

# We need to change something in the initial transformations, 
# leave it like this for now.
# now its hardcode but we will change it to be dynamic in the future
# """
# def randomfun(x):
#     processed = x
#     for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
#         processed[col] = processed[col].astype(object)
#     return processed
# prepro_train = randomfun(prepro_train)

# #---------------------------------get the X train and y train--------------------------------
# X_train = prepro_train.drop("readmitted", axis=1)
# y_train = prepro_train["readmitted"].copy() 

# #---------------------------------get the X test and y test--------------------------------
# #in a df mode not a numpy array
# prepro_test = preprocessing.transform(test_set)
# X_test = prepro_test.drop("readmitted", axis=1)
# y_test = prepro_test["readmitted"].copy()

# X_train = randomfun(X_train)
# X_test = randomfun(X_test)

# """
# In order to implament the data into ML model, we need to convert the data into
# data that can goes into onehotencoder and standard scaler for the categorical and
# standard scaler for the numerical data
# to do that we will use the col_processor pipeline which we wrote in the DefPipeLineClasses.py
# on the X_train and y_train to get the X_train_np and y_train_np
# """
# # --------------------------------- get the X train and y train --------------------------------
# # in a numpy array mode not a df
# X_train_np = col_processor.fit_transform(X_train)
# X_test_np = col_processor.transform(X_test)
# #%%
# # if we wish to see the data in a df mode we can use the following code
# def show_data():
#     print("X_train_np:")
#     display(pd.DataFrame(X_train_np.toarray()))
#     print("X_test_np:")
#     display(pd.DataFrame(X_test_np.toarray()))


# def main():
#     show_data()

# #to call the main function remove the comment from the next line
# # main()


# #%%
# col_processor.get_feature_names_out()



# ###########################
# # Set numerical columns
# num_cols = ['num_medications', 'num_lab_procedures']
# bin_cols = ['change', 'diabetesMed']
# # Set categorical columns
# cols = train_set.columns
# label = 'readmitted'
# cat_cols = [col for col in cols if col not in num_cols and col not in columns_to_drop and col not in label and col not in bin_cols]
