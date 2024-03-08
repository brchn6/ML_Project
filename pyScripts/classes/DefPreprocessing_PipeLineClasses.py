#%%
"""
This script is non callable (no main) and is used to create a pipeline object for the data preparation.
in this script we will create a pipeline objects and define the hard coded values we will use in the pipeline.
becuase the pipeline is not callable, we dont need to define the df as a parameter.
"""
#---------------------------------Libraries--------------------------------
#importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#%%
#defining the DropDup class
class DropDup(BaseEstimator, TransformerMixin):
    """Transformer class to drop duplicate rows based on a subset of columns.

    Parameters:
    subset_col (list): List of column names to consider for identifying duplicates.

    Returns:
    pandas.DataFrame: Transformed DataFrame with duplicate rows dropped.
    """
    def __init__(self, subset_col):
        self.subset_col = subset_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.drop_duplicates(subset=self.subset_col, keep="first").reset_index(drop=True)
        return X_transformed 

# Drop columns 
class DropColumns(BaseEstimator, TransformerMixin):
    """Transformer class to drop specified columns from a DataFrame.

    Parameters:
    columns_to_drop (list): List of column names to drop.

    Returns:
    pandas.DataFrame: Transformed DataFrame with specified columns dropped.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.drop(columns=self.columns_to_drop, errors='ignore')
        return X_transformed


class LabelFetcher(BaseEstimator, TransformerMixin):
    """
    A custom transformer class that fetches the labels column from the input data and returns the modified dataset without the labels column.
    """

    def __init__ (self):
        pass

    def fit(self, X, y=None):
        """
        Fit method required by scikit-learn's BaseEstimator.

        Parameters:
        - X: Input data (pandas DataFrame)

        Returns:
        - self: Returns the instance of the transformer
        """
        return self

    def transform(self, X):
        """
        Transform method required by scikit-learn's TransformerMixin.

        Parameters:
        - X: Input data (pandas DataFrame)

        Returns:
        - diabetes_labels: The labels column extracted from the input data
        - train_set_mod: The modified dataset without the labels column
        """
        # get the labels column
        diabetes_labels = X['readmitted']
        # Drop the label column
        train_set_mod = X.drop('readmitted', axis=1)
        return diabetes_labels, train_set_mod


# DiseaseConverter
class DiseaseConverter(BaseEstimator, TransformerMixin):
    """Transformer class to convert disease codes to their corresponding names.

    Returns:
    pandas.DataFrame: Transformed DataFrame with disease codes converted to names.
    """
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

#IDS transformer
class IDSTransformer(BaseEstimator, TransformerMixin):
    """Transformer class to transform IDS columns.

    Returns:
    pandas.DataFrame: Transformed DataFrame with IDS columns transformed.
    """
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
    

#Regrouping A1Cresult transformer 
#
class A1CTransformer(TransformerMixin):
    """Transformer class to regroup A1Cresult values.

    Returns:
    pandas.DataFrame: Transformed DataFrame with regrouped A1Cresult values.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Define your conditions here or make sure they are accessible
        cond1 = X['A1Cresult'].isna()
        cond2 = X['A1Cresult'].isin(['Norm'])
        cond3 = (X['A1Cresult'].isin(['>7'])) & (X['change'] == "No")
        cond4 = (X['A1Cresult'].isin(['>7'])) & (X['change'] == "Ch")
        cond5 = (X['A1Cresult'].isin(['>8'])) & (X['change'] == "No")
        cond6 = (X['A1Cresult'].isin(['>8'])) & (X['change'] == "Ch")

        # Create a copy of the DataFrame to modify
        X_transformed = X.copy()

        # Update values based on conditions
        X_transformed.loc[cond1, 'A1Cresult'] = 'No HbA1c test performed'
        X_transformed.loc[cond2, 'A1Cresult'] = 'HbA1c in normal range'
        X_transformed.loc[cond3, 'A1Cresult'] = 'HbA1c greater than 7%, but no med change'
        X_transformed.loc[cond4, 'A1Cresult'] = 'HbA1c greater than 7%, with med change'
        X_transformed.loc[cond5, 'A1Cresult'] = 'HbA1c greater than 8%, but no med change'
        X_transformed.loc[cond6, 'A1Cresult'] = 'HbA1c greater than 8%, with med change'

        return X_transformed

#Genral colomn regrouping:
class CustomTransformer(BaseEstimator, TransformerMixin):
    """Transformer class to apply custom functions to specified columns.

    Parameters:
    functions (dict): Dictionary mapping column names to corresponding functions.

    Returns:
    pandas.DataFrame: Transformed DataFrame with custom functions applied to specified columns.
    """
    def __init__(self, functions):
        self.functions = functions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, func in self.functions.items():
            X_transformed[col] = X_transformed[col].apply(func)
        return X_transformed
    
class CategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        object_columns = X.select_dtypes(include=['category']).columns
        if len(object_columns) == 0:
            return X
        else:
            X[object_columns] = X[object_columns].astype('object')
            return X

class BooleanConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].replace({'Yes': 1, 'Ch': 1, 'No': 0, 'Other': 0})
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features
