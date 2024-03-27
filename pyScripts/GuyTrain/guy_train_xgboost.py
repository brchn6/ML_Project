
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV

#set wd to be pyScripts:
#os.chdir(os.path.join(os.getcwd(), 'pyScripts'))

path = os.path.join(os.getcwd())

#load the data:
X_train = pd.read_csv(os.path.join(path, 'X_train_df.csv'))
X_test = pd.read_csv(os.path.join(path, 'X_test_df.csv'))
y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(path, 'y_test.csv'))

#convert the train and test set to DMatrix:
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

#save the DMatrix to a file:
dtrain.save_binary('dtrain.buffer')
dtest.save_binary('dtest.buffer')



"""
#load all the npy files
#dtest = xgb.DMatrix('dtest.buffer')
#dtrain = xgb.DMatrix('dtrain.buffer')


#Set random seed using NumPy
np.random.seed(42)

# Its optimal value highly depends on the other parameters, 
#and thus it should be re-tuned each time you update a parameter. 
# Number of boosting rounds to try
num_boost_round = 1000

#Splitting data into train and validation set:
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

#convert the train and validation set to DMatrix:
dtrain = xgb.DMatrix(X_train, label = y_train)
dval = xgb.DMatrix(X_val, label = y_val)
evals = [(dtrain, 'train'), (dval, 'eval')]

params = {
'objective': 'reg:logistic',
'eval_metric': 'logloss',
'eta': 0.1}

# Create a watchlist for early stopping
evals = [(dtrain, 'train'), (dval, 'eval')]

# Train the model
model = xgb.train(params=params,
                  dtrain=dtrain,
                  num_boost_round=num_boost_round,
                  evals=evals,
                  early_stopping_rounds=10,
                  verbose_eval=True
                  ) 

best_logloss = model.best_score
best_iteration = model.best_iteration + 1


#print the best logloss and best iteration:
print(f'Best logloss: {best_logloss}, best iteration: {best_iteration}')

"""

#Hyperparameter tuning:
num_boost_round = 1000

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:logistic',
}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=10,
    metrics={'logloss'},
    early_stopping_rounds=10
)

#Get the best logloss and best iteration:
best_logloss = cv_results['test-logloss-mean'].min()
best_iteration = len(cv_results)

"""
## Tuning max_depth and min_child_weight:
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(1,50,5)
    for min_child_weight in range(1,50,5)
]
"""

def cv_xgboost(params, dtrain, num_boost_round, gridsearch_params, par1, par2):
    min_logloss = float("Inf")
    best_params = None
    
    for param_1, param_2 in gridsearch_params:
        print("CV with {}={}, {}={}".format(par1,
                                            param_1,
                                            par2,
                                            param_2))
        # Update our parameters
        params[par1] = param_1
        params[par2] = param_2
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'logloss'},
            early_stopping_rounds=10
        )

        # Update best logloss
        mean_logloss = cv_results['test-logloss-mean'].min()
        boost_rounds = cv_results['test-logloss-mean'].argmin()
        print("\tLogloss {} for {} rounds".format(mean_logloss, boost_rounds))
        if mean_logloss < min_logloss:
            min_logloss = mean_logloss
            best_params = (param_1,param_2)
    print("Best params: {}, {}, Logloss: {}".format(best_params[0], best_params[1], min_logloss))



#############Results#############:

#Best params: 6, 6, Logloss: 0.20175233051126334

#Update params:

params['max_depth'] = 6
params['min_child_weight'] = 6

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(4,11)]
    for colsample in [i/10. for i in range(4,11)]
]

cv_xgboost(params, dtrain, num_boost_round, gridsearch_params, 'subsample', 'colsample_bytree')

#############Results#############:

#Best params: 1.0, 0.9, Logloss: 0.2016876811388658

#Update params:

params['subsample'] = 1.0
params['colsample_bytree'] = 0.9

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['logloss'],
            early_stopping_rounds=10
          )
        # Update best logloss
    mean_logloss = cv_results['test-logloss-mean'].min()
    boost_rounds = cv_results['test-logloss-mean'].argmin()
    print("\tLogloss {} for {} rounds".format(mean_logloss, boost_rounds))
    if mean_logloss < min_logloss:
        min_logloss = mean_logloss
        best_params = (param_1,param_2)
print("Best params: {}, {}, Logloss: {}".format(best_params[0], best_params[1], min_logloss))




