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
from sklearn.model_selection import GridSearchCV  ,StratifiedKFold  ,cross_validate
from sklearn.dummy import DummyClassifier
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
param_grid = make_grid_Parameters(RunParasSearch=True)

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



#---------------------------- fit the model with best hyperparameters ------------------------------
best_params = rf_model.best_params_
print('Best hyperparameters:', best_params)
logging.info('Best hyperparameters: %s' %best_params)
rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_train_es, y_train_es)
logging.info('besst model fitteed % s' %rf_best)
predictions = rf_best.predict_proba(X_val)
log_loss4 = log_loss(y_val, predictions)
print('logloss4 - best hyperparameter tuning with sklearn', log_loss4)
logging.info('logloss4 - best hyperparameter tuning with sklearn: %s' %log_loss4)


#---------------------------- # Feature Importance ------------------------------
feature_importances = rf_best.feature_importances_
print('Feature Importance:', feature_importances)
logging.info('Feature Importance: %s' %feature_importances)


#---------------------------- genarate 15 random seeds and calculate the mean log loss ------------------------------
# Generating prediction and feature importance tables on 15 different seeds:
prediction_table = pd.DataFrame()
scores = [log_loss, roc_auc_score]
scores_cm = [precision_score, recall_score, accuracy_score]

feature_importance_table = pd.DataFrame()

for i in range(15):
    best_model = RandomForestClassifier(**best_params, random_state=i)
    best_model.fit(X_train, y_train)
    preds_test = best_model.predict_proba(X_test)
    preds_cm = best_model.predict(X_test)

    # Storing scores in the prediction table
    for score in scores:
        prediction_table.loc['seed_' + str(i), score.__name__] = score(y_test, preds_test[:, 1])
    for score in scores_cm:
        prediction_table.loc['seed_' + str(i), score.__name__] = score(y_test, preds_cm)
    
    # Storing feature importances
    feature_importance_table.loc['seed_' + str(i)] = best_model.feature_importances_

# Export the prediction table to a CSV file
prediction_table.to_csv('prediction_table.csv')

# Export feature importance data
feature_importance_table.to_csv('feature_importance.csv')

# Optional: Create and save a feature importance plot (requires additional plotting libraries)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
feature_importance_table.mean().sort_values(ascending=False).plot(kind='bar', ax=ax)
plt.title('Average Feature Importances Across Seeds')
plt.savefig('feature_importances.png')

##---------------------------- Dummy Classifier and Default Model Comparison ------------------------------
# Dummy classifier for baseline comparison
strategies = ['most_frequent', 'uniform', 'constant']
constants = [None, None, 1]  # constant value only used with 'constant' strategy

for strategy, constant in zip(strategies, constants):
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42, constant=constant)
    dummy_clf.fit(X_train, y_train)
    dummy_pred = dummy_clf.predict(X_test)
    dummy_proba = dummy_clf.predict_proba(X_test)
    print('Strategy used:', strategy)
    for score in scores:
        print(score.__name__, round(score(y_test, dummy_proba[:, 1]), 5))
    for score in scores_cm:
        print(score.__name__, round(score(y_test, dummy_pred), 5))

# Comparing the best model with a default parameters model
def_params = {'n_estimators': 100, 'max_depth': None}  # Default parameters for RandomForest

default_rf = RandomForestClassifier(**def_params, random_state=42)
default_rf.fit(X_train, y_train)
default_pred_proba = default_rf.predict_proba(X_test)
for score in scores:
    print(score.__name__ + ' - default params', round(score(y_test, default_pred_proba[:, 1]), 5))


print("Training Time: %s seconds" % (str(time.time() - start_time)))
logging.info("Training Time: %s seconds" % (str(time.time() - start_time)))
