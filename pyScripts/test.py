#%%
import pandas as pd
import numpy as np
import os
import matplotlib as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

train_set = pd.read_csv(os.getcwd() + '/../data/train_set_test.csv')
train_set_mod = train_set.copy()

train_set_mod = train_set_mod.drop('readmitted', axis=1) 
diabetes_labels = train_set['readmitted'].copy()

from sklearn.base import BaseEstimator, TransformerMixin
from disease_ids_conds import *
#%%
#Drop duplicates
class DropDup(BaseEstimator, TransformerMixin):
    def __init__(self, subset_col):
        self.subset_col = subset_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.drop_duplicates(subset=self.subset_col, keep="first").reset_index(drop=True)
        return X_transformed 

dropdup_col = "patient_nbr"
#dup_dropper = DropDup(dropdup_col)

#df_dropdup = dup_dropper.transform(train_set_mod)

# Drop columns 
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.drop(columns=self.columns_to_drop, errors='ignore')
        return X_transformed

# Test DropColumns:
# Create an instance of the DropColumns transformer, specifying columns to drop
# Dropping columns found with large amount of missing values or one unique value
# Dropping patient number column - uninformative
# Should be explained in the EDA
columns_to_drop = ['payer_code', 'encounter_id', 'weight', 'patient_nbr', 'medical_specialty'] + ['acetohexamide', 'troglitazone', 'examide', 'citoglipton', 'metformin-rosiglitazone']
#column_dropper = DropColumns(columns_to_drop)

# Transform the DataFrame using the custom transformer
#df_dropped = column_dropper.transform(df_dropdup)

#Label column should be dropped after ther dimensions of the df is set.

class SplitLabels(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y =None):
        return self
    
    def transform(self, X):
        X_transformed = X.drop('readmitted', axis=1) 
        X_labels = X['readmitted'].copy()
        return X_transformed, X_labels

#labels_splitter = SplitLabels()
#df_dropped, diabetes_labels = labels_splitter.transform(df_dropped)

class DiseaseConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def convert_disease(self, value):

        if value == '?' or value == '789':
            return diseases[9]
        try:
            numeric_value = float(value)
        except ValueError:
            return diseases[9]  # Skip non-numeric values

        for id, disease in zip(ids, diseases):
            if numeric_value in id:
                return disease  # Replace with the string of your choosing
        return value   
    
    def transform(self, X):
        diag_columns = ['diag_1', 'diag_2', 'diag_3']
        X_transformed = X.copy()
        for col in diag_columns:
            X_transformed[col] = X_transformed[col].apply(self.convert_disease)
        return X_transformed

# Test DiseaseConverter:
# Create an instance of the DiseaseConverter transformer
#converter = DiseaseConverter()

# Transform the DataFrame using the custom transformer
#df_transformed = converter.transform(df_dropped)

Mapdf = pd.read_csv(os.getcwd() + '/../data/IDS_mapping.csv')

class IDSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Removing Dead patients:
        discharge_disposition_id_DF = Mapdf[["discharge_disposition_id", "description.1"]]
        discharge_disposition_id_DF = discharge_disposition_id_DF[discharge_disposition_id_DF['description.1'].str.contains('Hospice') | discharge_disposition_id_DF['description.1'].str.contains('Expired')]
        X_transformed = X[~X['discharge_disposition_id'].isin(discharge_disposition_id_DF ['discharge_disposition_id'])]
        #Regrouping IDS columns:
        X_transformed.loc[:, 'admission_type_id'] = X['admission_type_id'].replace([8, 6], 5).replace([7], 6)
        X_transformed.loc[:, 'discharge_disposition_id'] = X['discharge_disposition_id'].replace([list(range(3, 6)) + [10, 15, 9, 23, 24, 22] + list(range(27, 31))], 2).replace([6, 8], 3).replace(7, 4).replace(12, 5).replace([16, 17], 6).replace([25, 26, 18], 7)
        X_transformed.loc[:, 'admission_source_id'] = X['admission_source_id'].replace([2, 3], 1).replace([25, 22, 18, 19, 10, 5, 6, 7, 4], 2).replace(8, 3).replace([19, 20, 17, 15, 9], 4).replace([23, 24, 11, 12, 13, 14], 5)
        return X_transformed
    

#ids_change = IDSTransformer()

#df_ids = ids_change.transform(df_transformed)

# %%
#Regrouping A1Cresult transformer

class A1CTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()

        X_transformed.loc[cond1, 'A1Cresult'] = 'No HbA1c test performed'
        X_transformed.loc[cond2, 'A1Cresult'] = 'HbA1c in normal range'
        X_transformed.loc[cond3, 'A1Cresult'] = 'HbA1c greater than 7%, but no med change'
        X_transformed.loc[cond4, 'A1Cresult'] = 'HbA1c greater than 7%, with med change'
        X_transformed.loc[cond5, 'A1Cresult'] = 'HbA1c greater than 8%, but no med change'
        X_transformed.loc[cond6, 'A1Cresult'] = 'HbA1c greater than 8%, with med change'
        return X_transformed
    
#a1c_change = A1CTransformer()

#df_a1c = a1c_change.transform(df_ids)
# %%
#Genral colomn regrouping:

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, functions):
        self.functions = functions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, func in self.functions.items():
            X_transformed[col] = X_transformed[col].apply(func)
        return X_transformed

# Create an instance of the CustomTransformer with the functions dictionary
#custom_transformer = CustomTransformer(functions)

# Apply the transformer to the DataFrame
#df_regroup = custom_transformer.transform(df_a1c)

