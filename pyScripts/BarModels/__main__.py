# """
# Main file for the BarModels directory
# """
#---------------------------- Imports basic -------------------------------
import sys
import os
import logging
import time

import pandas as pd
import numpy as np


import warnings
warnings.filterwarnings('ignore')
startTime = time.time()
root = os.path.dirname(os.path.abspath(__name__))
              # root = '/home/labs/mayalab/barc/MSc_studies/ML_Project'
if root not in sys.path:
    sys.path.append(root)
else:
    pass

if __name__ == "__main__":
    here = '/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels'
else:
    here = os.path.dirname(os.path.abspath(__name__))
    

# %reload_ext autoreload
# %autoreload 2
#---------------------------- Imports for the model -------------------------------
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_defultParams
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_useingGridSearchCV
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_useing_Optuna

#---------------------------- Logger -------------------------------
def initialize_logging():
    log_directory = os.path.join(here, "logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(filename=os.path.join(log_directory, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"),
                         level=logging.INFO,
                         format="%(asctime)s - %(levelname)s - %(message)s")

def log_message(message):
    current_time = time.localtime() 
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logging.info(f"{time_string} - {message}")
    logging.getLogger().handlers[0].flush()

# Initialize logging
initialize_logging()
# Log a message
log_message(f" the logger is working, the time is: {time.time()-startTime}")
log_message(f"starting to load the data, the time is: {time.time()-startTime}")
# ---------------------------- data incoming -------------------------------
def load_data():
    # Load the data
    X_train = pd.read_csv(os.path.join(here, 'X_train_df.csv'))
    y_train = pd.read_csv(os.path.join(here, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(here, 'X_test_df.csv'))
    y_test = pd.read_csv(os.path.join(here, 'y_test.csv'))

    # Compare the columns of X_train and X_test
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    extra_cols = train_cols - test_cols

    # Remove any identified extra columns from X_train
    if extra_cols:
        X_train = X_train.drop(columns=list(extra_cols))

    # Convert DataFrame to numpy arrays and flatten
    X_train_np = X_train.values  # Converts DataFrame to numpy array
    y_train_np = y_train.values.ravel()  # Converts DataFrame to numpy array and flattens it
    X_test_np = X_test.values  # Converts DataFrame to numpy array
    y_test_np = y_test.values.ravel()  # Converts DataFrame to numpy array and flattens it

    return X_train_np, y_train_np, X_test_np, y_test_np
try :
    X_train_np, y_train, X_test_np, y_test = load_data()
    print(f"the data was loaded successfully, the shapes are: X_train: {X_train_np.shape}, y_train: {y_train.shape}, X_test: {X_test_np.shape}, y_test: {y_test.shape}")
except:
    raise Exception("the load_data function failed")
log_message(f"finished loading the data, the time is: {time.time()-startTime}")
# ---------------------------- Rendom_forest -------------------------------
log_message(f"starting to run the Rendom_forest model,the time is: {time.time()-startTime}")
def run_Rendom_forest():
    #create a Rendom_forest_classification_BC object using the default parameters
    rf = Rendom_forest_classification_BC_defultParams(X_train_np, y_train, X_test_np, y_test)
    #build the model
    classifier = rf.build_RandomForestClassifier()[0]
    classifier_fit = rf.build_RandomForestClassifier()[1]
    predictions_On_TrainDS = rf.predict_RandomForestClassifierTrainData(classifier_fit)
    predictions_On_TestDS = rf.predict_RandomForestClassifierTestData(classifier_fit)
    predictions_On_TrainDS_proba = rf.predict_RandomForestClassifierTrainData_proba(classifier_fit)
    predictions_On_TestDS_proba = rf.predict_RandomForestClassifierTestData_proba(classifier_fit)
    
    accuracy, f1_weighted, f1_binary = rf.accuracy_score(predictions_On_TrainDS, y_train)
    print(f"the results on the train data are: accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_binary: {f1_binary}")
    accuracy, f1_weighted, f1_binary = rf.accuracy_score(predictions_On_TestDS, y_test)
    print(f"the results on the test data are: accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_binary: {f1_binary}")


    log_loss, roc_auc = rf.accuracy_score_proba(predictions_On_TrainDS_proba, y_train)
    print(f"the results on the train data are: log_loss: {log_loss}, roc_auc: {roc_auc}")
    log_loss, roc_auc = rf.accuracy_score_proba(predictions_On_TestDS_proba, y_test)
    print(f"the results on the test data are: log_loss: {log_loss}, roc_auc: {roc_auc}")

    params = rf.get_params(classifier)

    confu = rf.make_confusion_matrix(predictions_On_TestDS, y_test)

    return classifier_fit, params , confu
log_message(f"finished running the Rendom_forest model,the time is: {time.time()-startTime}")
try :
    classifier_fit, params , confu = run_Rendom_forest()
    print(f"the best parameters are: {params}")
except:
    raise Exception("the run_Rendom_forest function failed")

#---------------------------- Rendom_forest with GridSearchCV -------------------------------
def run_Rendom_forest_with_GridSearchCV():
    #create a Rendom_forest_classification_BC instance using the GridSearchCV
    rf_GS_CV = Rendom_forest_classification_BC_useingGridSearchCV(X_train_np, y_train, X_test_np, y_test)
    
    #build the model using the GridSearchCV
    classifier, classifier_fit, best_rf_classifier = rf_GS_CV.build_RandomForestClassifierWithGridSearchCV()
    
    #update the parameter grid
    updated_param_grid = rf_GS_CV.update_parameter_grid()

    #get the best parameters
    best_params = rf_GS_CV.get_best_params()
    print(f"the best parameters are: {best_params}")

    predictions_On_TrainDS = rf_GS_CV.predict_RandomForestClassifier(y_train)[0]
    predictions_On_TestDS = rf_GS_CV.predict_RandomForestClassifier(y_test)[1]
    predictions_On_TrainDS_proba = rf_GS_CV.predict_RandomForestClassifier(y_train)[2]
    predictions_On_TestDS_proba = rf_GS_CV.predict_RandomForestClassifier(y_test)[3]

    accuracy, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TrainDS, y_train)[0:3]
    print(f"the results on the train data are: accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_binary: {f1_binary}")
    log_loss, roc_auc = rf_GS_CV.accuracy_score(predictions_On_TrainDS_proba, y_train)[3:5]
    print(f"the results on the train data are: log_loss: {log_loss}, roc_auc: {roc_auc}")
    
    accuracy, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TestDS, y_test)[0:3]
    print(f"the results on the test data are: accuracy: {accuracy}, f1_weighted: {f1_weighted}, f1_binary: {f1_binary}")
    log_loss, roc_auc = rf_GS_CV.accuracy_score(predictions_On_TestDS_proba, y_test)[3:5]
    print(f"the results on the test data are: log_loss: {log_loss}, roc_auc: {roc_auc}")
    
    params = rf_GS_CV.get_best_params()
    confu = rf_GS_CV.make_confusion_matrix(predictions_On_TestDS, y_test)

    
    return classifier_fit, best_rf_classifier , updated_param_grid
try:
    classifier_fit, best_rf_classifier , updated_param_grid , params , confu = run_Rendom_forest_with_GridSearchCV()
except:
    raise Exception("the run_Rendom_forest_with_GridSearchCV function failed")

# The best parameters in this run are: {'bootstrap': False, 'max_depth': 20, 'max_features': 'sqrt',
# 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 200} and the ROC AUC score is: 0.9222283046173643



# def Runrf_using_Optuna():
#     rf_model = Rendom_forest_classification_BC_useing_Optuna(X_train_np, y_train, X_test_np, y_test)
#     best_params = rf_model.build_RandomForestClassifierWithOptuna()[2]
#     best_classifier = rf_model.build_RandomForestClassifierWithOptuna()[0]
#     rf_model.evaluate_model(best_classifier)

#     return best_params , best_classifier , rf_model
# best_params = Runrf_using_Optuna()