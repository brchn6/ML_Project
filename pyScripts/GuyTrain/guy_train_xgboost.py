#%%
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from scipy.sparse import csr_matrix

path = os.path.join(os.getcwd())

#load all the npy files
X_train = np.load(path + '/X_train_np.npy', allow_pickle=True).item()
y_train = np.load(path + '/y_train.npy', allow_pickle=True)
X_test = np.load(path + '/X_test_np.npy', allow_pickle=True).item()
y_test = np.load(path + '/y_test.npy',  allow_pickle=True)



#%%
dtrain = xgb.DMatrix(X_train, label = y_train)



#%%
#Define scoring methods:



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
