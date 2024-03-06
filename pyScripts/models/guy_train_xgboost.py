#%%
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from preprocessing_pipe_copy import *

#Set up GAN-balanced diabetes dataframe and separate from labels:
X_train = pd.read_csv('balanced_train_set.csv')
y_train = X_train.readmitted.copy()
X_train = X_train.drop('readmitted', axis=1)

#Process dataframe through pipeline:
X_train = col_processor_s.fit_transform(X_train)

#Define scoring methods:

scoring = {
    'neg_log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr') 
}

for i in range(2):
    #Set random seed:
    seed = i*42
    np.random.seed(seed)
    
    #Define xbg classifier
    xgb_clf = XGBClassifier(random_state=seed)

    #Early stopping section, 1st try:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

    model = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=2)

    





    


# %%
