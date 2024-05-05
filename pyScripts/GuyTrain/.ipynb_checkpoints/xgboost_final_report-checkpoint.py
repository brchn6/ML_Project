#%%
import numpy as np 
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, f1_score, recall_score

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

    def_params = {}

    best_params = { 
                     "max_depth" : 14,
                     "learning_rate" : 0.09996208924063056,
                     "n_estimators" : 71,
                     "subsample" : 0.01713632457120349,
                     "gamma" : 4.280323790629931e-08,
                     "booster" : 'gbtree',
                     "min_child_weight" : 22, 
                     "lambda" : 0.0016148099053915346,
                     "alpha": 0.2324766684035494,
                     "scale_pos_weight" : 0.9072249881289736,
                     "max_delta_step" : 0,                     
                     "grow_policy" : "depthwise",
                     'colsample_bytree' : 0.03461865397687917
                }

    dtrain = xgb.DMatrix(X_train, label = y_train, enable_categorical=True) 
        
    #Obtaining score for default parameters:  
    xgb_cv = xgb.cv(dtrain=dtrain, params=def_params, nfold=10, metrics='logloss', seed=42, verbose_eval=True) 
    print('logloss_default - no hyperparameter tunning', xgb_cv['test-logloss-mean'].iloc[-1])

    #Obtaining results after hyperparameters tuning with best params:
    xgb_cv = xgb.cv(dtrain=dtrain, params=best_params, nfold=10, metrics='logloss',seed=42, verbose_eval=True) 
    print('logloss_final - number of estimators with cv', xgb_cv['test-logloss-mean'].iloc[-1])

    #Score from test_set prediction:
    prediction_table = pd.DataFrame()
    scores = [log_loss, roc_auc_score]

    for i in range(15):
        best_model = xgb.XGBClassifier(**best_params, random_state=i, enable_categorical=True, tree_method='hist')
        best_model.fit(X_train, y_train)
        preds_test = best_model.predict_proba(X_test)
        for score in scores:
            prediction_table.loc['seed_'+str(i), score.__name__] = score(y_test, preds_test[:,1])
