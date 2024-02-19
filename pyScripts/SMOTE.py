#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
import os
import sys
from AddRootDirectoriesToSysPath import add_directories_to_sys
add_directories_to_sys(os.getcwd())

from ScriptNumberOne import train_set
from Run_pipeline import *

#add the data from scrinum1
data = train_set.copy()

#assign the target and the features
y_train = diabetes_labels
X_train = diabetes_test

## make sure you ran script number two that includes the new function to_categorical that converts
## dtype object to categorical. otherwise you will get an error.

# the order of the pipeline should be as describe below:
# step 1: smote, step 2: pre-processing, step 3: the classifier with default parameters.
# the classifiers should be defined before as classes.

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as imbpipeline

classifiers = [XGBClassifier, LGBMClassifier, CatBoostClassifier, SVC, BalancedRandomForestClassifier]
score = ['neg_log_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
sm = SMOTENC(random_state=42, categorical_features=cat_cols)

for classifier in classifiers:
    for s in score:
        pipeline = imbpipeline(steps=[
            ['smote', sm],
            ["classifier", classifier()]
        ])
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=s)
    
        # Store cross validation scores
        # Add your code here to store cross validation scores for each iteration
