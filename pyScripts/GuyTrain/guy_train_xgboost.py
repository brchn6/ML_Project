
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

#load all the npy files
#dtest = xgb.DMatrix('dtest.buffer')
#dtrain = xgb.DMatrix('dtrain.buffer')

evals_result = {}

# Its optimal value highly depends on the other parameters, 
#and thus it should be re-tuned each time you update a parameter. 
# Number of boosting rounds to try
num_boost_round = 1000

# List of random seeds to use
seeds = range(15)  # Using 15 seeds

testsize = 1/len(seeds)

score_tree_table = pd.DataFrame(columns = ['seed', 'best_logloss', 'best_num_trees'])

for seed in seeds:

    print(f"\nTesting with random seed: {seed}")

    testsize = len(seeds)

    #Splitting data into train and validation set:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = testsize, random_state = seed)

    #convert the train and validation set to DMatrix:
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dval = xgb.DMatrix(X_val, label = y_val)

    evals = [(dtrain, 'train'), (dval, 'eval')]

    # Set random seed using NumPy
    np.random.seed(seed)

    params = {
    'objective': 'reg:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1}

    # Create a watchlist for early stopping
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    # Train the model
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=num_boost_round,
                      evals=watchlist,
                      early_stopping_rounds=100,
                      verbose_eval=True
                      ) 
    
    best_iteration = model.best_iteration
    best_logloss = model.best_score

    # Update the score_tree_table
    new_row = {'seed': seed, 'best_logloss': best_logloss, 'best_num_trees': best_iteration + 1}
    score_tree_table = pd.concat([score_tree_table, pd.DataFrame([new_row])], ignore_index=True)

    print("Best logloss: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))
    
#Export score_tree_table to csv
score_tree_table.to_csv(os.path.join(path, 'score_tree_table.csv'))

#save the model:
model.save_model(os.path.join(path, 'xgboost.model'))
    

    
    


# %%
