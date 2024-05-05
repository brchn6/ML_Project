
import tensorflow as tf
import numpy as np 
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, f1_score

#Check available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

# Specify GPU device
gpu_device = physical_devices[0]  # Assuming you have at least one GPU
tf.config.experimental.set_memory_growth(gpu_device, True)

path = os.path.join(os.getcwd())

X_train = pd.read_csv(os.path.join(path, 'X_train_df.csv'))
X_test = pd.read_csv(os.path.join(path, 'X_test_df.csv'))
y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(path, 'y_test.csv'))

columns = ['diag_3_365.44', 'repaglinide_Down']

def removeRogueColumns(df):
        df.drop(columns, axis=1, inplace=True)
        return df

X_train = removeRogueColumns(X_train)


if __name__ == "__main__":
    
        # choose if want to use set of communly used start values for the hyperparameters 
        commonly_used_start_values = True 
        print('commonly_used_start_values', commonly_used_start_values)

        if commonly_used_start_values == True: 
                params = {'device' : "cuda", 'max_depth':5, 'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8}
        else: 
                params = {}

        
        early_stopping_rounds=10

        X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)

        # DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed. You can construct DMatrix from multiple different sources of data.
        dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical=True) 
        
        
        # Obtaining the untuned logloss
        xgb_reg = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_cv = xgb.cv(dtrain=dtrain, params=params, nfold=10, metrics='logloss', seed=42, verbose_eval=True) 
        print('logloss1 - no hyperparameter tunning', xgb_cv['test-logloss-mean'].iloc[-1])
        
        # Early stopping to find the number of estimator
        xgb_reg = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds, **params, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_reg.fit(X_train_es, y_train_es,
                eval_set=[(X_val, y_val)], verbose=True, eval_metric='logloss')
        
        num_estimators  = xgb_reg.best_iteration
        print('num_estimators', num_estimators)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=True) 
        print('logloss2 - number of estimators with cv', xgb_cv['test-logloss-mean'].iloc[-1])
        
        params['n_estimators'] = num_estimators
 
        prediction_table = pd.DataFrame()
        scores = [log_loss, roc_auc_score]

        for i in range(15):
            best_model = xgb.XGBClassifier(**params, random_state=i, enable_categorical=True, tree_method='hist')
            best_model.fit(X_train, y_train)
            preds_test = best_model.predict_proba(X_test)
            for score in scores:
                prediction_table.loc['seed_'+str(i), score.__name__] = score(y_test, preds_test[:,1])

        #Export the prediction table to a csv file
        prediction_table.to_csv('prediction_table.csv') 

        if commonly_used_start_values == True: 
                xgb_reg = xgb.XGBClassifier(max_depth=5, subsample=0.8, gamma=0, colsample_bytree=0.8, random_state=42, enable_categorical=True, tree_method='hist')
        else: 
                xgb_reg = xgb.XGBClassifier(random_state=42, enable_categorical=True, tree_method='hist')
        
        xgb_reg.fit(X_train, y_train, verbose=False)
        y_pred = xgb_reg.predict(X_test)
        for score in scores:
            score_preds_default = score(y_pred, y_test)
            print('logloss-test - no hyperparameter tunning', score_preds_default)
            print("We have reduced logloss by ", score_preds_default - score(y_test, preds_test[:,1]))

