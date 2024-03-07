#%%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .DefPreprocessing_PipeLineClasses import BooleanConverter
# ---------------------------------Define the object for the colprocessor pipeline--------------------------------
"""
this object is used to in the script RunPipe.py to make the data ready to be used in the ML model
"""

# Define preprocessing steps for numerical and categorical columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

bool_transformer = Pipeline(steps=[
    ('booltransform', BooleanConverter())])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',drop='if_binary'))])

