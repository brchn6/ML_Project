import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import os
import numpy as np

os.chdir('C:\py_projects\ML')

train_notmod = pd.read_csv(r'train_set_saar.csv')
train_mod = pd.read_csv(r'train_set_disease_cat_saar.csv')

from sklearn.base import BaseEstimator, TransformerMixin

class DiseaseConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def convert_disease(self, value):
        
        diseases = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes','Diabetes Uncontrolled', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
        ids = ids = [
        (list(range(390, 460)) + [785]),
        (list(range(460, 520)) + [786]),
        (list(range(520, 580)) + [787]),
        ([250.00, 250.01]),
        np.round(np.arange(250.02, 251.00, 0.01), 2).tolist(),
        (list(range(800, 1000))),
        (list(range(710, 740))),
        (list(range(580, 630)) + [788]),
        (list(range(140, 240))),  
        (list(range(790, 800)) + 
        list(range(240, 250)) + 
        list(range(251, 280)) + 
        list(range(680, 710)) + 
        list(range(780, 785)) + 
        list(range(290, 320)) + 
        list(range(280, 290)) + 
        list(range(320, 360)) + 
        list(range(630, 680)) + 
        list(range(360, 390)) + 
        list(range(740, 760)) +
        list(range(1,140)))
            ]

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
converter = DiseaseConverter()

# Transform the DataFrame using the custom transformer
df_transformed = converter.transform(train_notmod)

# Display the original and transformed DataFrames
print("Original DataFrame:")
print(train_notmod)
print("\nTransformed DataFrame:")
print(df_transformed)
print("Manually transformed")
print(train_mod)

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
columns_to_drop = ['payer_code', 'encounter_id']
column_dropper = DropColumns(columns_to_drop)

# Transform the DataFrame using the custom transformer
df_dropped = column_dropper.transform(train_notmod)

# Display the original and transformed DataFrames
print("Original DataFrame:")
print(train_notmod.columns)

print("\nTransformed DataFrame:")
print(df_dropped.columns)


