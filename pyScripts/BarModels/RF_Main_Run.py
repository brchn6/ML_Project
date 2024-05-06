#%%
# Description: This file is the main file for running the Random Forest model. It imports the necessary libraries and modules, and sets up the logger. It also imports the necessary libraries for the model and sets up the logger.
#---------------------------- Imports -------------------------------
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import os
import sys

here= os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.abspath(__name__))
if root not in sys.path:
    sys.path.append(root)

import logging
import time
start_time = time.time()
#---------------------------- initialize_logger -------------------------------
def initialize_logger(log_file="logfile%s.log" %time.strftime("%Y%m%d%H%M%S")):
    """Initialize logger."""
    log_directory = os.path.join(here, 'logs', str("logs%s" %__file__.split('/')[-1].split('.')[0]))
    try:
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        raise

    log_file_path = os.path.join(log_directory, log_file)
    
    try:
        logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', 
                            datefmt='%d-%b-%y %H:%M:%S')
    except Exception as e:
        print(f"Error setting up logger configuration: {e}")
        raise
    return logging.getLogger()
initialize_logger()
logging = logging.getLogger(__name__)
logging.info('Logger initialized')
#---------------------------- Logger Done -------------------------------
#---------------------------- Imports for the model -------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss , f1_score ,accuracy_score, roc_auc_score, precision_score, recall_score , confusion_matrix ,mean_absolute_error, mean_squared_error, median_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV ,StratifiedKFold , cross_val_score ,cross_validate
#---------------------------- function to import data -------------------------------
def import_data(name):
    """Import data."""
    data = pd.read_csv(os.path.join(here, name))
    return data

#---------------------------- import data -------------------------------
try:
    X_train = import_data('X_train_df.csv')
    y_train = import_data('y_train.csv')
    X_test = import_data('X_test_df.csv')
    y_test = import_data('y_test.csv')
except Exception as e:
    print(f"Error importing data: {e}")
    raise

#---------------------------- DropColumns -------------------------------
try:
    columns = ['diag_3_365.44', 'repaglinide_Down']
    X_train = X_train.drop(columns, axis=1)
except Exception as e:
    print(f"Error dropping columns: {e}")
    raise

#---------------------------- Splitting the train set into train and validation set in 80:20 -------------------------------
try:
    X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=42)
except Exception as e:
    print(f"Error splitting the train set: {e}")
    raise

#---------------------------- create a GridSearchCV object -------------------------------
def make_grid_Parameters(RunParasSearch=False):
    if RunParasSearch:
        # Define the parameter grid
        params = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'criterion': ['gini', 'entropy']
        }
    else:
        params = {
            'bootstrap': [False],  # Wrap single value in a list
            'max_depth': [20],  # Wrap single value in a list
            'max_features': ['sqrt'],  # Wrap single value in a list
            'min_samples_leaf': [2],  # Wrap single value in a list
            'min_samples_split': [20],  # Wrap single value in a list
            'n_estimators': [200]  # Wrap single value in a list
        }
    return params
param_grid = make_grid_Parameters(RunParasSearch=False)

#---------------------------- create a Random Forest model -------------------------------
def create_model(random_state=42, n_jobs=-1, early_stopping=None, **param_grid):
    model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **param_grid, verbose=0)
    return model
rf_model = create_model()

#---------------------------- create a CV and GridSearchCV object ---------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_grid_search = GridSearchCV(estimator= rf_model, param_grid= param_grid, cv=cv, scoring='neg_log_loss', n_jobs=-1, verbose=2)
#---------------------------- fit the model no hyperparameter tuning with sklearn -------------------------------
rf_model.fit(X_train_es, y_train_es)
#---------------------------- predict the model -------------------------------
def predict_model(model, X_val):
    y_pred = model.predict(X_val)
    rf_pre_Proba= model.predict_proba(X_val)    
    return y_pred , rf_pre_Proba

y_pred, rf_pre_Proba = predict_model(rf_model, X_val)
log_loss1 = log_loss(y_val, rf_pre_Proba)
print('logloss1 - no hyperparameter tuning with sklearn', log_loss1)
logging.info('logloss1 - no hyperparameter tuning with sklearn: %s' %log_loss1)
# Perform cross-validation
cv_results = cross_validate(rf_model, X_train, y_train, cv=5, scoring='neg_log_loss')
print('logloss1 - no hyperparameter tunning with sk_cv', cv_results['test_score'].mean())
logging.info('logloss1 - no hyperparameter tunning with sk_cv: %s' %cv_results['test_score'].mean())

#---------------------------- fit the model with early stopping ------------------------------
# Early stopping to find the number of estimators:
rf_model = create_model(early_stopping=50)
rf_model.fit(X_train_es, y_train_es)
y_pred, rf_pre_Proba = predict_model(rf_model, X_val)
logloss2 = log_loss(y_val, rf_pre_Proba)
print('logloss2 - early stopping with sklearn', logloss2)
logging.info('logloss2 - early stopping with sklearn: %s' %logloss2)
# Perform cross-validation
cv_results = cross_validate(rf_model, X_train, y_train, cv=5, scoring='neg_log_loss')
print('logloss2 - no hyperparameter tunning with sk_cv early stopping with sklearn', cv_results['test_score'].mean())
logging.info('logloss2 - no hyperparameter tunning with sk_cv early stopping with sklearn: %s' %cv_results['test_score'].mean())


#---------------------------- fit the model with hyperparameter tuning ------------------------------
rf_grid_search.fit(X_train_es, y_train_es)
#---------------------------- predict the model ------------------------------
y_pred, rf_pre_Proba = predict_model(rf_grid_search, X_val)
logloss3 = log_loss(y_val, rf_pre_Proba)
print('logloss3 - hyperparameter tuning with sklearn', logloss3)
logging.info('logloss3 - hyperparameter tuning with sklearn: %s' %logloss3)
# Perform cross-validation
cv_results = cross_validate(rf_grid_search, X_train, y_train, cv=5, scoring='neg_log_loss')
print('logloss3 - hyperparameter tunning with sk_cv', cv_results['test_score'].mean())
logging.info('logloss3 - hyperparameter tunning with sk_cv: %s' %cv_results['test_score'].mean())


#---------------------------- fit the model with hyperparameter tuning using RandomizedSearchCV ------------------------------
rf_model= create_model()
rf_random_search = RandomizedSearchCV(estimator= rf_model, param_distributions= param_grid, n_iter=100, cv=cv, scoring='neg_log_loss', n_jobs=-1, verbose=2)
rf_random_search.fit(X_train_es, y_train_es)
best_params = rf_random_search.best_params_
#---------------------------- predict the model ------------------------------
y_pred, rf_pre_Proba = predict_model(rf_random_search, X_val)
logloss4 = log_loss(y_val, rf_pre_Proba)
print('logloss4 - hyperparameter tuning with RandomizedSearchCV', logloss4)
logging.info('logloss4 - hyperparameter tuning with RandomizedSearchCV: %s' %logloss4)
# Perform cross-validation
cv_results = cross_validate(rf_random_search, X_train, y_train, cv=5, scoring='neg_log_loss')
print('logloss4 - hyperparameter tunning with sk_cv', cv_results['test_score'].mean())
logging.info('logloss4 - hyperparameter tunning with sk_cv: %s' %cv_results['test_score'].mean())

# Repeating the process 15 times with random seed
log_losses = []
for i in range(15):
    # Set a new random seed
    random_seed = 42 + i
    # Initialize RandomForestClassifier with the new random seed
    rf_model = RandomForestClassifier(random_state=random_seed, **best_params)
    # Fit the model
    rf_model.fit(X_train, y_train)
    # Predict probabilities
    y_pred_proba = rf_model.predict_proba(X_val)
    # Calculate log loss
    log_loss_val = log_loss(y_val, y_pred_proba)
    log_losses.append(log_loss_val)
    print(f'Log loss for iteration {i+1}: {log_loss_val}')
    logging.info(f'Log loss for iteration {i+1}: {log_loss_val}')
# Calculate mean log loss
mean_log_loss = np.mean(log_losses)
print(f'Mean log loss: {mean_log_loss}')
logging.info(f'Mean log loss: {mean_log_loss}')


logging.info('Done')
logging.info(f'Execution time: {time.time() - start_time} seconds')
#---------------------------- Done ------------------------------



