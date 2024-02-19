#%%
# ------------------------------ code ------------------------------ #
import sys
import os
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.pipeline import Pipeline

# Import all from Run_pipeline module
from Run_pipeline import *

# Load the dataset
X_train = diabetes_test
y_train = diabetes_labels

# def function to convert object columns to categorical
def to_categorical(data):
    object_columns = data.select_dtypes(include=['object']).columns
    data[object_columns] = data[object_columns].astype('category')  
    return data
#%%

X_train= to_categorical(X_train)

# print(y_train.value_counts())
# Assuming 'NO' and '<30' are negative outcomes, and '>=30' is a positive outcome
y_train = y_train = y_train.map({'NO': 0, '<30': 1})
# print(y_train.value_counts())

# Define the classifiers to be evaluated
classifiers = [XGBClassifier, LGBMClassifier, CatBoostClassifier, SVC, BalancedRandomForestClassifier]

# Define the scoring metrics
score = ['neg_log_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Initialize SMOTENC for handling categorical features
sm = SMOTENC(random_state=42, categorical_features='auto')

# Initialize cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create an empty dictionary to store cross-validation scores
cv_scores = {}

# Iterate over classifiers
for classifier in classifiers:
    # Iterate over scoring metrics
    for s in score:
        # Define the pipeline
        pipeline = Pipeline(steps=[
            ['smote', sm],
            ["classifier", classifier()]
        ])

        # Perform cross-validation
        cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=s)

        # Store cross-validation scores
        classifier_name = classifier.__name__  # Get the name of the classifier
        if classifier_name not in cv_scores:
            cv_scores[classifier_name] = {}
        cv_scores[classifier_name][s] = cross_val_scores

# Now, cv_scores dictionary contains cross-validation scores for each classifier and each scoring metric
# You can analyze or further process these scores as needed
        
#data output as tbl
import pandas as pd
df = pd.DataFrame(cv_scores)
df.to_csv('cv_scores.csv')
# ------------------------------ end ------------------------------ #
#%%
import os
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())

# Import all from Run_pipeline module
from Run_pipeline import *

# Load the dataset
X = diabetes_test
y = diabetes_labels


# def function to convert object columns to categorical
def to_categorical(data):
    object_columns = data.select_dtypes(include=['object']).columns
    data[object_columns] = data[object_columns].astype('category')  
    return data

X = to_categorical(X)

from collections import Counter
from imblearn.over_sampling import SMOTENC
print(f'Original dataset shape {X.shape}')
print(f'Original dataset samples per class {Counter(y)}')
sm = SMOTENC(random_state=42, categorical_features='auto')
X_res, y_res = sm.fit_resample(X, y)
print(f'Resampled dataset samples per class {Counter(y_res)}')


#%%

#plot pie chart of before
import matplotlib.pyplot as plt
import numpy as np


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
data = Counter(y)
labels = list(data.keys())
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Original dataset samples per class')
plt.show()


# #plot pie chart of after
# labels = ['NO', '<30', '>30']
# sizes = [54864, 11357, 35545]
# explode = (0, 0, 0.1)  # only "explode" the 3rd slice (i.e. '>=30')
# plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Original dataset samples per class')
# plt.show()
