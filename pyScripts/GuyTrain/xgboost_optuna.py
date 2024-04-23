#This script was to try another approach to hyperparameter tuning using Optuna.
#The script was not used to get the final scores for the model, but it was a good practice to try another approach to hyperparameter tuning. 

#Importing the necessary libraries:
import time
import numpy as np 
import optuna 
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

#Setting main path:
path = os.path.join(os.getcwd())

#Reading prepared train and test sets (from main.py script):
X_train = pd.read_csv(os.path.join(path, 'X_train_df.csv'))
X_test = pd.read_csv(os.path.join(path, 'X_test_df.csv'))
y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(path, 'y_test.csv'))

#Removing columns that were present in the train_set but not in the test_set
#This happend due to onehotencoding and extremely rare values that only went to train set:
columns = ['diag_3_365.44', 'repaglinide_Down']

def removeRogueColumns(df):
        df.drop(columns, axis=1, inplace=True)
        return df

X_train = removeRogueColumns(X_train)

#Define optuna objective function:
def objective(trial, dtrain, params_in_stages): # categorical_feats
                  
        params = { 
                     "max_depth" : trial.suggest_int("max_depth", 1, 20),
                     "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.1),
                     "n_estimators" : trial.suggest_int('n_estimators', 5, 300),
                     'subsample': trial.suggest_float('subsample', 0.01, 1.0, log = True),
                     'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
                     "booster" : trial.suggest_categorical('booster', ['gbtree', 'dart']),
                     "min_child_weight" : trial.suggest_int("min_child_weight", 1, 50), 
                     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log = True),
                     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log = True),
                     "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 0.5, 1),
                     "max_delta_step" : trial.suggest_int("max_delta_step", 0, 10),                     
                     "grow_policy" : trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log = True),

                }

        #update the params with the params_in_stages after each trail:
        params.update(params_in_stages)

        #Get the number of estimators from the previous stage:
        num_boost_round = params.pop('n_estimators')

        #Run native XGBoost CV:
        xgb_cv = xgb.cv(params=params, num_boost_round=num_boost_round, dtrain=dtrain, nfold=10, shuffle=True, stratified=True, metrics="logloss", seed=42) 
    
        #Get the best logloss score:
        score = xgb_cv['test-logloss-mean'].iloc[-1]
        print(f"Trial {trial.number}:, Score: {score}")
        return score

if __name__ == "__main__":

        start = time.time()
    
        # choose if want to use set of communly used start values for the hyperparameters 
        commonly_used_start_values = True 
        print('commonly_used_start_values', commonly_used_start_values)

        if commonly_used_start_values == True: 
                params = {'device' : "cuda", 'max_depth':5, 'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8}
        else: 
                params = {}

        
        n_trials = 10000
        early_stopping_rounds=10

        X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)

        # DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed. 
        dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical=True) 
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Obtaining the untuned logloss
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_cv1 = xgb.cv(dtrain=dtrain, params=params, nfold=10, metrics='logloss', seed=42, verbose_eval=False, shuffle=True, stratified=True) 
        print('logloss1 - no hyperparameter tunning native', xgb_cv1['test-logloss-mean'].min())

        xgb_cv_try = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring='neg_log_loss')
        print('logloss1 - no hyperparameter tunning with sk_cv', xgb_cv_try['test_score'].mean())
        
        # Early stopping to find the number of estimator:
        xgb_clf = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds, **params, random_state=42, enable_categorical=True)
        xgb_clf.fit(X_train_es, y_train_es,
                eval_set=[(X_val, y_val)], verbose=False, eval_metric='logloss')
        
        # Get the number of estimators:
        num_estimators  = xgb_clf.best_iteration
        print('num_estimators', num_estimators)

        xgb_cv2 = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True) 
        print('logloss2 - number of estimators with cv native', xgb_cv2['test-logloss-mean'].min())

        xgb_clf = xgb.XGBClassifier(**params, n_estimators = num_estimators, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_cv_try = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring='neg_log_loss')
        print('logloss2 - number of estimators with sk_cv', xgb_cv_try['test_score'].mean())
        
        params['n_estimators'] = num_estimators

        #Optuna hyperparameter tuning:
        study = optuna.create_study(direction='minimize')
        func = lambda trial: objective(trial, dtrain, params)
        study.optimize(func, n_trials= n_trials) 
    
        #Get the best params:
        best_params = study.best_params
        print(best_params)
        best_params.update(params)
        print("All Best params :", best_params)

        #Train the model with the best params:
        best_model = xgb.XGBClassifier(**best_params, random_state=42, enable_categorical=True)
        best_model.fit(X_train, y_train)   
        preds_test = best_model.predict_proba(X_test)
        print('logloss_final_hyperparameter_tuned_test', log_loss(y_test, preds_test[:,1]),
                                                         roc_auc_score(y_test, preds_test[:,1]))
        
        #Generarting prediction table and feature importance table on 15 different seeds:
        prediction_table = pd.DataFrame()
        scores = [log_loss, roc_auc_score]
        scores_cm = [precision_score, recall_score, accuracy_score]

        for i in range(15):
            best_model = xgb.XGBClassifier(**best_params, random_state=i, enable_categorical=True)
            best_model.fit(X_train, y_train)
            preds_test = best_model.predict_proba(X_test)
            preds_cm = best_model.predict(X_test)
            for score in scores:
                prediction_table.loc['seed_'+str(i), score.__name__] = score(y_test, preds_test[:,1])
        for score in scores_cm:
               prediction_table.loc['seed_'+str(i), score.__name__] = score(y_test, preds_cm)

        #Export the prediction table to a csv file
        prediction_table.to_csv('prediction_table.csv') 

        #Comparing the best model with default parameters:
        if commonly_used_start_values == True: 
                xgb_clf = xgb.XGBClassifier(max_depth=5, subsample=0.8, gamma=0, colsample_bytree=0.8, random_state=42, enable_categorical=True, tree_method='hist')
        else: 
                xgb_clf = xgb.XGBClassifier(random_state=42, enable_categorical=True)
        
        xgb_clf.fit(X_train, y_train, verbose=False)
        y_pred = xgb_clf.predict(X_test)
        for score in scores:
            score_preds_default = score(y_pred, y_test)
            print('logloss-test - no hyperparameter tunning', score_preds_default)
            print("We have reduced logloss by ", score_preds_default - score(y_test, preds_test[:,1]))

print("GPU Training Time: %s seconds" % (str(time.time() - start)))