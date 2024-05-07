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


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
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
logger = logging.getLogger(__name__)
logger.info('Logger initialized')
#---------------------------- Logger Done -------------------------------
#---------------------------- Imports for the model -------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss , f1_score ,accuracy_score, roc_auc_score, precision_score, recall_score , confusion_matrix ,mean_absolute_error, mean_squared_error, median_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV  ,StratifiedKFold  ,cross_validate
from sklearn.dummy import DummyClassifier
import seaborn as sns
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
except FileNotFoundError as e:
    logging.error("File not found: %s", e)
    sys.exit(1)
except pd.errors.EmptyDataError as e:
    logging.error("Empty data found while importing: %s", e)
    sys.exit(1)

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
def get_params(RunParasSearch=False):
    if RunParasSearch:
        # Define the parameter grid for GridSearchCV
        return {
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
        # Define a single set of parameters for direct model instantiation
        return {
            'bootstrap': False,
            'max_depth': 20,
            'max_features': 'sqrt',
            'min_samples_leaf': 2,
            'min_samples_split': 20,
            'n_estimators': 200
        }
params = get_params(RunParasSearch=False)
param_grid = get_params(RunParasSearch=True)
#---------------------------- create a Random Forest model -------------------------------
def create_model(params, random_state=42, n_jobs=-1):
    model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **params)
    return model
#---------------------------- create a CV object -------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            #---------------------------------the first model is trained with no hyperparameter tuning----------------
#---------------------------- predict the model -------------------------------
def predict_model(model, X_val):
    y_pred = model.predict(X_val)
    rf_pre_Proba= model.predict_proba(X_val)    
    log_lossval = log_loss(y_val, rf_pre_Proba)
    print('logloss :', log_lossval)
    logging.info('logloss : %s' %log_lossval)
    return y_pred , rf_pre_Proba , log_lossval

def Perform_cross_validation(model, X_train, y_train):
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring='neg_log_loss')
    print('logloss - no hyperparameter tunning with sk_cv', cv_results['test_score'].mean())
    logging.info('logloss - no hyperparameter tunning with sk_cv: %s' %cv_results['test_score'].mean())
    return cv_results['test_score'].mean()

def Get_best_hyperparameters(rf_grid_search):
    best_params = rf_grid_search.best_params_
    print('Best hyperparameters:', best_params)
    logging.info('Best hyperparameters: %s' %best_params)
    return best_params

def get_feature_importance(rf_best):
    feature_importances = rf_best.feature_importances_
    print('Feature Importance:', feature_importances)
    logging.info('Feature Importance: %s' %feature_importances)
    return feature_importances

#---------------------------- predict the model ------------------------------
try:
    #---------------------------- fit the model no hyperparameter tuning with sklearn -------------------------------
    rf_model = create_model(params)
    rf_model.fit(X_train_es, y_train_es)
    y_pred, rf_pre_Proba ,log_loss1 = predict_model(rf_model, X_val)
    # Perform cross-validation
    cv_results = Perform_cross_validation(rf_model, X_train, y_train)
    print("finished to fit the model with no hyperparameter tuning, time: %s seconds" % (str(time.time() - start_time)))
except Exception as e:
    print(f"Error fitting the model with no hyperparameter tuning: {e}")
    raise
               
               #---------------------------------the second model is trained with hyperparameter tuning----------------
#---------------------------- fit the model with hyperparameter tuning ------------------------------
try:
    rf_grid_search = GridSearchCV(estimator= rf_model, param_grid= param_grid, cv=cv, scoring='neg_log_loss', n_jobs=-1, verbose=2)
    rf_grid_search.fit(X_train_es, y_train_es)
    #---------------------------- predict the model ------------------------------
    y_pred, rf_pre_Proba ,log_loss2 = predict_model(rf_grid_search, X_val)

    # Perform cross-validation
    cv_results = Perform_cross_validation(rf_grid_search, X_train, y_train)

    #---------------------------- fit the model with best hyperparameters ------------------------------\
    #implement the Get_best_hyperparameters function
    best_params = Get_best_hyperparameters(rf_grid_search)
    print("finished to fit the model with the grid search, now ig goes to the best hyperparameters, time: %s seconds" % (str(time.time() - start_time)))
    print("best_params:", best_params)
except Exception as e:
    print(f"Error fitting the model with hyperparameter tuning: {e}")
    raise
            #---------------------------------the third model is trained with best hyperparameters----------------

try :
    #---------------------------- create a model with best hyperparameters ------------------------------
    rf_best = create_model(best_params)
    #---------------------------- fit the model with best hyperparameters ------------------------------
    rf_best.fit(X_train_es, y_train_es)
    #---------------------------- predict the model ------------------------------
    y_pred, rf_pre_Proba ,log_loss3 = predict_model(rf_best, X_val)
except Exception as e:
    print(f"Error fitting the model with best hyperparameters: {e}")
    raise

#---------------------------- # Feature Importance ------------------------------
feature_importances = get_feature_importance(rf_best)

#---------------------------- genarate 15 random seeds and calculate the mean log loss ------------------------------
def generate_prediction_and_feature_tables(X_train, y_train, X_test, y_test, best_params, scores=[log_loss, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score], scores_cm=[confusion_matrix]):
    # Generating prediction and feature importance tables on 15 different seeds:
    prediction_table = pd.DataFrame()
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

    try:
        if not os.path.exists(os.path.join(here, 'results')):
            os.makedirs(os.path.join(here, 'results'))
    except Exception as e:
        print(f"Error creating results directory: {e}")
        raise
    
    try:
        # Export the prediction table to a CSV file
        prediction_table.to_csv('results/prediction_table.csv')

        # Export feature importance data
        feature_importance_table.to_csv('results/feature_importance_table.csv')
    except Exception as e:
        print(f"Error exporting data: {e}")
        raise

    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=feature_importance_table.columns, y=feature_importance_table.mean(), ax=ax)
        ax.set_title('Feature Importance')
        ax.set_ylabel('Mean Feature Importance')
        ax.set_xlabel('Features')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        raise

# Call the function
generate_prediction_and_feature_tables(X_train, y_train, X_test, y_test, best_params)

##---------------------------- Dummy Classifier and Default Model Comparison ------------------------------
def dummy_classifier_comparison(X_train, y_train, X_test, y_test, scores=[log_loss, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score], scores_cm=[confusion_matrix]):
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

# Call the function
dummy_classifier_comparison(X_train, y_train, X_test, y_test)


#---------------------------- Model Evaluation ------------------------------