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
    print("finished to fit the model with no hyperparameter tuning, time: %s seconds" % (str(time.time() - start_time)))
except Exception as e:
    print(f"Error fitting the model with no hyperparameter tuning: {e}")
    raise
               
               #---------------------------------the second model is trained with hyperparameter tuning----------------
#---------------------------- fit the model with hyperparameter tuning ------------------------------
try:
    #---------------------------- fit the model with best hyperparameters ------------------------------\
    #implement the Get_best_hyperparameters function
    best_params = params
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
def generate_prediction_and_feature_tables(X_train, y_train, X_test, y_test, best_params, scores, scores_cm, scores_binary):
    # Generating prediction and feature importance tables on 15 different seeds:
    prediction_table = pd.DataFrame()
    feature_importance_table = pd.DataFrame(columns=[f'feature_{i}' for i in range(X_train.shape[1])])

    for i in range(15):
        best_model = RandomForestClassifier(random_state=i, **best_params)
        best_model.fit(X_train, y_train)
        preds_test = best_model.predict_proba(X_test)
        preds_cm = best_model.predict(X_test)

        # Storing scores in the prediction table
        try:
            for score in scores:
                prediction_table.loc['seed_' + str(i), score.__name__] = score(y_test, preds_test[:, 1])
            for score in scores_binary:
                prediction_table.loc['seed_' + str(i), score.__name__] = score(y_test, preds_cm)
            for score in scores_cm:
                prediction_table.loc['seed_' + str(i), 'confusion_matrix'] = str(score(y_test, preds_cm))
        except Exception as e:
            print(f"Error storing scores in the prediction table: {e}")
            raise

        # Storing feature importances
        try:
            feature_importance_table.loc['seed_' + str(i)] = best_model.feature_importances_
        except Exception as e:
            print(f"Error storing feature importances: {e}")
            raise
        
    try:
        if not os.path.exists(os.path.join(here, 'results')):
            os.makedirs(os.path.join(here, 'results'))
    except Exception as e:
        print(f"Error creating results directory: {e}")
        raise

    try:
        # Export the prediction table to a CSV file
        prediction_table.to_csv(os.path.join(here, 'results', 'prediction_table.csv'))

        # Export feature importance data
        feature_importance_table.to_csv(os.path.join(here, 'results', 'feature_importance_table.csv'))
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
        plt.savefig(os.path.join(here, 'results', 'feature_importance_plot.png'))
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        raise

##---------------------------- Dummy Classifier and Default Model Comparison ------------------------------
def dummy_classifier_comparison(X_train, y_train, X_test, y_test, scores, scores_cm, scores_binary):
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()

    # Dummy classifier for baseline comparison
    strategies = ['most_frequent', 'uniform', 'constant']
    constants = [None, None, 1]  # constant value only used with 'constant' strategy

    for strategy, constant in zip(strategies, constants):
        dummy_clf = DummyClassifier(strategy=strategy, constant=constant if strategy == 'constant' else None, random_state=42)
        dummy_clf.fit(X_train, y_train)
        dummy_pred = dummy_clf.predict(X_test)
        dummy_proba = dummy_clf.predict_proba(X_test)
        print('Strategy used:', strategy)
        logging.info('Strategy used: %s' %strategy)

        for score in scores:
            if hasattr(score, '__call__'):
                print(score.__name__, round(score(y_test, dummy_proba[:, 1]), 5))
        
        for score in scores_binary:
            if hasattr(score, '__call__'):
                print(score.__name__, round(score(y_test, dummy_pred), 5))

        if strategy != 'constant':  # Confusion matrix doesn't work well with constant if y_test doesn't contain '1'
            for score in scores_cm:
                print(score.__name__, confusion_matrix(y_test, dummy_pred).tolist())

    # Comparing the best model with default parameters model
    def_params = {'n_estimators': 100, 'max_depth': None}  # Default parameters for RandomForest
    default_rf = RandomForestClassifier(**def_params, random_state=42)
    default_rf.fit(X_train, y_train)
    default_pred_proba = default_rf.predict_proba(X_test)
    
    for score in scores:
        if hasattr(score, '__call__'):
            print(score.__name__ + ' - default params', round(score(y_test, default_pred_proba[:, 1]), 5))

#---------------------------- main function ------------------------------
def main():
    scores = [log_loss, roc_auc_score]  # Metrics that can handle probabilities
    scores_binary = [f1_score, accuracy_score, precision_score, recall_score]  # Metrics that need binary inputs
    scores_cm = [confusion_matrix]  # Metrics that use confusion matrix
    generate_prediction_and_feature_tables(X_train, y_train, X_test, y_test, best_params, scores, scores_cm, scores_binary)
    dummy_classifier_comparison(X_train, y_train, X_test, y_test, scores, scores_cm, scores_binary)
    # Timing and logging
    elapsed_time = time.time() - start_time
    print("Training Time: %s seconds" % elapsed_time)
    logging.info("Training Time: %s seconds" % elapsed_time)

# if __name__ == '__main__':
#     main()




#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
try:
    dir= "/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/results"
    feature_importance_table= pd.read_csv(os.path.join(dir, 'feature_importance_table.csv'))
    feature_importance_table= feature_importance_table.drop('Unnamed: 0', axis=1)
    # feature_importance_table = feature_importance_table.iloc[:, 0:15]
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create a bar plot
    mean_importances = feature_importance_table.mean()
    sns.barplot(x=mean_importances.index, 
                y=mean_importances.values, 
                ax=ax, 
                order=mean_importances.sort_values(ascending=False).index)
    # Additional plot formatting
    ax.set_title('Mean Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
    plt.tight_layout()
    plt.savefig(os.path.join(here, 'results', 'feature_importance_plot.png'))
except Exception as e:
    print(f"Error plotting feature importance: {e}")
    raise



#%%
def transform_feature_names(feature_names):
    # Function to return the feature name up to the last '_'
    def get_feature_name(name):
        return name[:name.rfind('_')]

    # Apply the get_feature_name function to all feature names
    feature_names = list(map(get_feature_name, feature_names))

    # Replace specific strings in the feature names
    replacements = {
        'age_Older': 'age',
        'admission_type_id_trauma': 'admission_type_id',
        'discharge_disposition_id_discharged': 'discharge_disposition_id',
        'discharge_disposition_id_home': 'discharge_disposition_id',
        'discharge_disposition_id_left': 'discharge_disposition_id',
        'admission_source_id_medical': 'admission_source_id',
        'num': 'num_medications',
        'num_lab': 'num_lab_procedures',
        'chang': 'change',
        'diabetesMe': 'diabetesMed'
    }
    feature_names = [replacements.get(name, name) for name in feature_names]

    return feature_names


def featureImportancePlot (feature_names, table):
    
    table_mean = table.mean(axis = 0)
    table_std = table.std(axis = 0)

    #Transform the feature names:
    feature_names = transform_feature_names(feature_names)

    #Change first column names:
    feature_names[0] = 'number_emergency'
    feature_names[1] = 'number_outpatient'
    
    feature_count = pd.DataFrame(feature_names).value_counts()
    
    sums = {}
    stds = {}
    for i in range(len(feature_names)):
        name = feature_names[i]
        if name in sums:
            sums[name] += table_mean[i]
            stds[name] += table_std[i]
        else:
            sums[name] = table_mean[i]
            stds[name] = table_std[i]
    
    #Divide the stds by the feature count:
    for key in stds:
        stds[key] = stds[key]/feature_count[key]
    
    #sort the sums in descending order:
    sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))
    
    #order stds by sums:
    stds = {k: stds[k] for k in sums.keys()}
    
    #plot the feature importances according to the sums dictionary (which is sorted):
    plt.figure(figsize=(10, 5))
    
    error = list(stds.values())[:10][::-1]
    plt.barh(list(sums.keys())[:10][::-1], list(sums.values())[:10][::-1], align='center', xerr=error, color='skyblue', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.title('Top 10 Feature Importance on (mean of 15 seeds)')
    plt.tight_layout()

X_train= pd.read_csv(os.path.join(here, 'X_train_df.csv'))
X_train = X_train.drop(columns=['diag_3_365.44', 'repaglinide_Down'], axis=1)
featureImportancePlot(X_train.columns, feature_importance_table)
plt.savefig(os.path.join(here, 'results', 'feature_importance_plot.png'))



#%%
#print all the parameters of the model
print(rf_best.get_params())



#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
dir= "/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/results"
prediction_table= pd.read_csv(os.path.join(dir, 'prediction_table.csv'))
prediction_table

def Print_Mean_of_15_seeds(prediction_table):
#    Neg log loss	Precision	Recall	ROC AUC	Accuracy
    def calcMean(x):
        return x.mean()
    def calcStd(x):
        return x.std()
    mean_values = prediction_table.apply(calcMean)
    std_values = prediction_table.apply(calcStd)

    # Create a new dataframe with the mean and standard deviation values
    result_table = pd.DataFrame({'Model': mean_values.index,
                                 'Neg log loss': f"{mean_values['log_loss']:.3f} ± {std_values['log_loss']:.3f}",
                                 'Precision': f"{mean_values['precision_score']:.3f} ± {std_values['precision_score']:.3f}",
                                 'Recall': f"{mean_values['recall_score']:.3f} ± {std_values['recall_score']:.3f}",
                                 'ROC AUC': f"{mean_values['roc_auc_score']:.3f} ± {std_values['roc_auc_score']:.3f}",
                                 'Accuracy': f"{mean_values['accuracy_score']:.4f} ± {std_values['accuracy_score']:.1E}"}, columns=columns)
    
    # Print the result table
    print(result_table)
    return result_table



Print_Mean_of_15_seeds(prediction_table.iloc[:, 1:-1])
