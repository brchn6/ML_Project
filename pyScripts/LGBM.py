# LGBM
import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform

#Plan:
# 1. Use default parameters to get the neg log loss on the training data and by cross validation.
#    This will be my reference point.
# 2. 

# Hyper-Parameter tuning:
# Step 1: optimizing number of trees 'n_estimators' early stopping.

# Import train data:

X_train = pd.read_csv('C:\Git\ML_project\pyScripts\GuyTrain\X_train_df.csv') 
y_train = pd.read_csv('C:\Git\ML_project\pyScripts\GuyTrain\y_train.csv') 

train_dataset = lgbm.Dataset(X_train, label=y_train)

# params = {'objective': 'binary',
#     'n_estimators': 95,
#     'max_depth': 10,
#     'num_leaves': 15,
#     'min_child_samples': 80,
#     'colsample_bytree': 1,
#     'subsample': 0,
#     'bagging_fraction': 0.5,
#     'subsample_freq': 0,
#     'reg_alpha': 0.2,
#     'reg_lambda': 0.1,
#     # 'min_split_gain':0,
#     # 'min_gain_to_split':0,
#     # 'min_child_weight':1,
#     'learning_rate':0.1,
#     'max_bin':255
#     }

params_lgb = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_boost_round' : 95,
            'subsample':0,
            'subsample_freq':0 
}


# divide the train set to train and validation in order to fix the number of trees.
X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True,random_state=42)

# lgbm_es = lgbm.LGBMClassifier(**params, callbacks= [lgbm.early_stopping(stopping_rounds = 10000)],random_state=42)
# lgbm_es.fit(X_train_es, y_train_es, eval_set=[(X_val, y_val)]) - Consider removing

train_d = lgbm.Dataset(X_train_es, label=y_train_es)
valid_d = lgbm.Dataset(X_val, label=y_val)
lr_range = [0.0001,0.001,0.01,0.1,0.2,0.5]
for lr in lr_range:
    params_es = {'objective': 'binary','learning_rate': lr}
    lgbm_es = lgbm.train(params=params_es,train_set=train_d, valid_sets=[valid_d], callbacks=[lgbm.early_stopping(stopping_rounds=10)])
    num_estimators = lgbm_es.best_iteration
    params_es['n_estimators']=num_estimators
    print('learning_rate: ',lr)
    print('best_iteration: ',lgbm_es.best_iteration)
    
    # Best learning rate is 0.1, it dictates n_estimators = 95. with a valid_0 loss = 0.200864.

# Evaluate untuned model on the train set:
params_train = {'objective': 'binary',
        'learning_rate': 0.1,  
        num_estimators : 95
}

train_all = lgbm.Dataset(X_train, label=y_train)
lgbm_train_p = lgbm.train(params=params_train,train_set=train_all)
y_pred_train_prob = lgbm_train_p.predict(X_train)
neg_log_loss_train = -log_loss(y_train, y_pred_train_prob)
print("Negative log loss on training set:", neg_log_loss_train)
# Negative log loss on training set: -0.17860238301838519. with params_train

# ## Base CV score, after fixing n_estimators = 95.

# # base_cv = lgbm.cv(params = params_lgb,train_set=train_dataset,nfold=10,metrics='binary_logloss',num_boost_round=95)
# # cv_logloss = cross_val_score(estimator, X_train, y_train, cv=10, scoring='neg_log_loss')
# # mean_logloss = np.mean(cv_logloss)
# # std_logloss = np.std(cv_logloss)
# # print("Mean neg log loss:", mean_logloss)
# # print("STD neg log loss:", std_logloss)

# # log_losses = base_cv['binary_logloss-mean']
# # print('Mean Log Loss:',np.mean(log_losses))
# # print('Standard Deviation Log Loss:',np.std(log_losses))

# # Hyper-Parameters tuning:
params = {'objective': 'binary',
        'learning_rate': 0.1,  
        'num_estimators' : 95,
        'boosting_type': 'dart',
        'max_depth': 7,
        'num_leaves': 20,
        'bagging_fraction': 0.2, 
        'colsample_bytree': 0.4,
        'min_child_samples': 60,
        'subsample': 0,
        'subsample_freq': 0
}

params_grid_a = {
                'boosting_type':['gbdt','dart','rf'],
                'max_depth': range(3,20,1),
                'num_leaves': [5, 20, 80, 320]
            } # neg_log_loss = -0.28, best: {'boosting_type': 'dart', 'max_depth': 7, 'num_leaves': 20}

params_grid_b = {
    'subsample':np.arange(0,1.2,0.2),
    'colsample_bytree':np.arange(0,1.2,0.2),
    'subsample_freq': np.arange(0,1.2,0.2),
    'bagging_fraction': np.arange(0,1.2,0.2),
    'min_child_samples': range(0,100,20)
} # neg_log_loss -0.27994427248028686, best: {'bagging_fraction': 0.2, 'colsample_bytree': 0.4, 'min_child_samples': 60, 'subsample': 0, 'subsample_freq': 0}

params_grid_c = {   
    'reg_alpha': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10],
    'reg_lambda': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10] 
                }

params_grid_d = {'learning_rate':np.arange(0,1,0.001),
                'feature_fraction':np.arange(0,1,0.2),
                'bagging_freq':np.arange(1,10,1)}

params_grid_e = {
                'is_unbalance':['True','False'],
                'scale_pos_weight': np.arange(1,10,1)
                }

# ## Hyper-parameter tuning using GridSearchCV

# grids = ['params_grid_a','params_grid_b','params_grid_c','params_grid_d','params_grid_e']
grids = [params_grid_c,params_grid_d,params_grid_e]

for grid in grids:
    estimator = lgbm.LGBMClassifier(**params, random_state=42)
    gsearch = GridSearchCV(param_grid=grid, estimator=estimator,
    scoring='neg_log_loss', cv=10)
    gsearch.fit(X_train,y_train)
    print(gsearch.best_params_)
    print('neg_log_loss', gsearch.best_score_)
    params.update(gsearch.best_params_)

print(params)

# # Create a RandomizedSearchCV Object

# params_grid_random = {
#     'min_split_gain':randint(0,1),
#     'min_gain_to_split':randint(0,1),
#     'min_child_weight': randint(0,100),
#     'max_bin': randint(10, 1000),
#     'min_sum_hessian_in_leaf': randint(1e-5,10)

# random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=50, scoring='neg_log_loss', cv=10,random_state=42)

# # Fit RandomizedSearchCV Object
# random_search.fit(X_train, y_train)

# print("Best Parameters found: ", random_search.best_params_)
# print("Best Score found: ", random_search.best_score_)
# params.update(random_search.best_params_)

# params = {'objective': 'binary',
#     'n_estimators': 95,
#     'max_depth': 10,
#     'num_leaves': 15,
#     'min_child_samples': 80,
#     'colsample_bytree': 1,
#     'subsample': 0,
#     'bagging_fraction': 0.5,
#     'subsample_freq': 0,
#     'reg_alpha': 0.2,
#     'reg_lambda': 0.1,
#     'min_split_gain':0,
#     'min_gain_to_split':0,
#     'min_child_weight':1,
#     'learning_rate':0.1,
#     'max_bin':255
#     }

# and improved the -log loss from -0.328 to -0.2844 (it changed 'gbdt' to 'dart')

# Predict over 15 random states:

# for i in range(1,16):
#     estimator.predict()

