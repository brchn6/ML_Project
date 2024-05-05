#Importing the necessary libraries and scripts from the pyScripts folder:
import time
import numpy as np 
import optuna 
import pandas as pd
import xgboost as xgb
import cupy as cp
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from feature_importance_script import *
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, precision_score, recall_score, accuracy_score

#Removing annoying warnings:
import warnings
warnings.filterwarnings('ignore')

#Setting main path:
path = os.path.join(os.getcwd()) + '/pyScripts/GuyTrain/'

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

#Calling main function:
if __name__ == "__main__":

        start = time.time()
    
        # choose if want to use set of communly used start values for the hyperparameters 
        commonly_used_start_values = True 
        print('commonly_used_start_values', commonly_used_start_values)

        if commonly_used_start_values == True: 
                params = {'device' : "cuda", 
                          'max_depth':5, 
                          'subsample':0.8, 
                          'gamma':0, 
                          'colsample_bytree':0.8,
                          'objective' : 'binary:logistic'}
        else: 
                params = {}
        #Setting the early stopping rounds to find best number of estimators:
        early_stopping_rounds=50
        #Splitting the train set into train and validation sets:
        X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)

        # DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed. 
        #Setting the DMatrix for the train set:
        dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical=True) 

        #Will use stratisfied cross validation in gridsearchcv, this will match our results to the initial results obtained under default parameters.
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Obtaining the untuned logloss score with both sklearn and native xgboost cv, this process will repeat after each hyperparameter tunning step:
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_cv = xgb.cv(dtrain=dtrain, params=params, nfold=10, metrics='logloss', seed=42, verbose_eval=False, shuffle=True, stratified=True) 
        print('logloss1 - no hyperparameter tunning native', xgb_cv['test-logloss-mean'].iloc[-1])

        xgb_cv_sk = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring='neg_log_loss')
        print('logloss1 - no hyperparameter tunning with sk_cv', xgb_cv_sk['test_score'].mean())
        
        # Early stopping to find the number of estimator:
        xgb_clf = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds, **params, random_state=42, enable_categorical=True, n_estimators=1000)
        xgb_clf.fit(X_train_es, y_train_es,
                eval_set=[(X_val, y_val)], verbose=False, eval_metric='logloss')
        
        #Saving the best number of estimators:
        num_estimators  = xgb_clf.best_iteration
        print('num_estimators', num_estimators)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True) 
        print('logloss2 - number of estimators native', xgb_cv['test-logloss-mean'].iloc[-1])

        params['n_estimators'] = num_estimators

        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')
        xgb_cv_sk = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring='neg_log_loss')
        print('logloss2 - number of estimators with sk_cv', xgb_cv_sk['test_score'].mean())

        #Hyperparameter tuning:
        #Tuning max_depth and min_child_weight:
        param_test1 = {
                'max_depth':range(3,20,2),
                'min_child_weight':range(1,40,2)
        }

        gsearch1 = GridSearchCV(param_grid=param_test1, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)  
        gsearch1.fit(X_train,y_train)
        print(gsearch1.best_params_) 
        print('logloss3 - max_depth and min_child_weight with gridsearchcv', abs(gsearch1.best_score_))

        #Params should be updated before running the native xgboost cv:
        #Updating the best parameters:
        params.update(gsearch1.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss3 - max_depth and min_child_weight native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Tuning gamma:
        param_test2= {'gamma':np.arange(0, 5, 0.01)}
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist') 

        gsearch2 = GridSearchCV(param_grid=param_test2, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)
        gsearch2.fit(X_train,y_train)
        print(gsearch2.best_params_) 
        print('logloss4 - gamma with gridsearchcv', abs(gsearch2.best_score_))

        #Updating the best parameters:
        params.update(gsearch2.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss4 - gamma native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Tuning subsample, colsample_bytree and colsample_bylevel:
        param_test3= {'subsample': np.linspace(1, 0.1, 10),
                      'colsample_bytree': np.linspace(1, 0.1, 10),
                      'colsample_bylevel': np.linspace(1, 0.1, 10)}
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist') 

        gsearch3 = GridSearchCV(param_grid=param_test3, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)
        gsearch3.fit(X_train,y_train)
        print(gsearch3.best_params_) 
        print('logloss5 - subsample, colsample_bytree and colsample_bylevel a with gridsearchcv', abs(gsearch3.best_score_))

        #Updating the best parameters:
        params.update(gsearch3.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss5 - subsample, colsample_bytree and colsample_bylevel native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Tuning reg_alpha and reg_lambda:
        param_test4 = {'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 1000, 0.001, 0.005, 0.01, 0.05],
                       'reg_lambda' : [1e-5, 1e-2, 0.1, 1, 1000, 0.001, 0.005, 0.01, 0.05]}
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')

        gsearch4 = GridSearchCV(param_grid=param_test4, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)
        gsearch4.fit(X_train,y_train)
        print(gsearch4.best_params_) 
        print('logloss6 - reg_alpha and reg_lambda with gridsearchcv', abs(gsearch4.best_score_))

        #Updating the best parameters:
        params.update(gsearch4.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss6 - reg_alpha and reg_lambda native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Tuning scale_pos_weight, max_delta_step, grow_policy and booster:
        param_test5 = {'scale_pos_weight':np.linspace(1,0.1, 10),
                       "max_delta_step" :  range(0,10,1),
                       "grow_policy" : ["depthwise", "lossguide"],
                       "booster" : ["gbtree", "dart"]}
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')               

        gsearch5= GridSearchCV(param_grid=param_test5, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)
        gsearch5.fit(X_train,y_train)
        print(gsearch5.best_params_) 
        print('logloss7 - scale_pos_weight, max_delta_step, grow_policy, booster with gridsearchcv', abs(gsearch5.best_score_))

        #Updating the best parameters:
        params.update(gsearch5.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = num_estimators, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss7 - scale_pos_weight, max_delta_step, grow_policy, booster native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Tuning learning_rate:
        lr = {'learning_rate':np.linspace(0.1,0.01, 10)}
        xgb_clf = xgb.XGBClassifier(**params, random_state=42, enable_categorical=True, tree_method='hist')

        gsearch6= GridSearchCV(param_grid=lr, estimator=xgb_clf, scoring='neg_log_loss', cv=cv)
        gsearch6.fit(X_train,y_train)
        print(gsearch6.best_params_) 
        print('logloss8 - learning_rate with gridsearchcv', abs(gsearch6.best_score_))

        #Updating the best parameters:
        params.update(gsearch6.best_params_)

        xgb_cv = xgb.cv(dtrain=dtrain, params=params, num_boost_round = 5000, nfold=10, metrics='logloss',seed=42, verbose_eval=False, shuffle=True, stratified=True)
        print('logloss8 - learning_rate native', xgb_cv['test-logloss-mean'].iloc[-1])

        #Updating the best number of estimators:
        final_num_estimators  = xgb_cv['test-logloss-mean'].idxmin()
        params['n_estimators'] = final_num_estimators

        best_params = params

        #print the best parameters after tunning:
        print('best parameters after tunning:', best_params)

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

        #Get feature names and feature importance table:
        feature_names = best_model.feature_names_in_
        fi_table = pd.DataFrame(columns=feature_names)

        for i in range(15):
            best_model = xgb.XGBClassifier(**best_params, random_state=i, enable_categorical=True, tree_method='hist')
            best_model.fit(X_train, y_train)
            fi_table.loc['seed_'+str(i)] = best_model.feature_importances_

        #Generate fi_plot:
        fi_plot = featureImportancePlot(feature_names, fi_table)
        
        #Export fi_plot to a png file:
        fi_plot.savefig('fi_plot.png')

        #Dummy classifier on 3 different strategies:
        strategies = ['most_frequent', 'uniform', 'constant']
        constants = [None, None, 1] 

        for strategy, constant in zip(strategies, constants):
                dummy_clf = DummyClassifier(random_state=42, strategy=strategy, constant=constant)
                dummy_clf.fit(X_train, y_train)
                dummy_pred = dummy_clf.predict(X_test)
                dummy_proba = dummy_clf.predict_proba(X_test)
                print('Strategy used: ', strategy)
                for score in scores:
                        print(score.__name__, round(score(y_test, dummy_proba[:,1]),5))
                for score in scores_cm:
                        print(score.__name__, round(score(y_test, dummy_pred),5))

        #Comparing the best model with default parameters:
        def_params = {'device' : "cuda", 
                          'max_depth':5, 
                          'subsample':0.8, 
                          'gamma':0, 
                          'colsample_bytree':0.8,
                          'objective' : 'binary:logistic'}

        if commonly_used_start_values == True: 
                xgb_reg = xgb.XGBClassifier(**def_params, random_state=42, enable_categorical=True, tree_method='hist')
        else: 
                xgb_reg = xgb.XGBClassifier(random_state=42, enable_categorical=True, tree_method='hist')
        
        xgb_reg.fit(X_train, y_train, verbose=False)
        y_pred = xgb_reg.predict_proba(X_test)
        for score in scores:
            score_preds_default = score(y_pred, y_test)
            print('logloss-test - no hyperparameter tunning', score_preds_default)
            print("We have reduced logloss by ", score_preds_default - score(y_test, preds_test[:,1]))   

        print("Training Time: %s seconds" % (str(time.time() - start)))