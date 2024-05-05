# """
# Main file for the BarModels directory
# """
#---------------------------- Imports basic -------------------------------
import sys
import os
import logging
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
startTime = time.time()
root = os.path.dirname(os.path.abspath(__name__))
              # root = '/home/labs/mayalab/barc/MSc_studies/ML_Project'
if root not in sys.path:
    sys.path.append(root)
else:
    pass
here=os.path.dirname(os.path.abspath(__file__))
              # here = '/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels'

# %reload_ext autoreload
# %autoreload 2
#---------------------------- Imports for the model -------------------------------
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_defultParams
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_useingGridSearchCV

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
    #load the data:
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

    X_train_np=X_train
    X_test_np=X_test
    return X_train_np, y_train, X_test_np, y_test

X_train_np, y_train, X_test_np, y_test = load_data()
log_message(f"finished loading the data, the time is: {time.time()-startTime}")
# ---------------------------- Rendom_forest -------------------------------
log_message(f"starting to run the Rendom_forest model,the time is: {time.time()-startTime}")
def run_Rendom_forest():
    #create a Rendom_forest_classification_BC object using the default parameters
    rf = Rendom_forest_classification_BC_defultParams(X_train_np, y_train, X_test_np, y_test)
    #build the model
    classifier_fit = rf.build_RandomForestClassifier()
    #predict the model
    predictions_On_TrainDS = rf.predict_RandomForestClassifierTrainData(classifier_fit)
    predictions_On_TestDS = rf.predict_RandomForestClassifierTestData(classifier_fit)
    #check the accuracy base on accuracy_score log_loss and f1_score
    accuracy , logLoss, f1_weighted, f1_binary = rf.accuracy_score(predictions_On_TrainDS, y_train)
    print ("accuracy: ", accuracy
        , "logLoss: ", logLoss
        , "f1_weighted: ", f1_weighted
        , "f1_binary: ", f1_binary)

    data_to_log = {"accuracy": accuracy, "logLoss": logLoss, "f1_weighted": f1_weighted, "f1_binary": f1_binary}
    log_message(f"the parameters that we got from the default parameters are: {classifier_fit.get_params()}, the time is: {time.time()-startTime}")
    log_message(f"now we are after the run of the Rendom_forest model with the default parameters,and the results on the train data are: {data_to_log}, the time is: {time.time()-startTime}")

    accuracy , logLoss, f1_weighted, f1_binary = rf.accuracy_score(predictions_On_TestDS, y_test)
    print ("accuracy: ", accuracy
        , "logLoss: ", logLoss
        , "f1_weighted: ", f1_weighted
        , "f1_binary: ", f1_binary)

    data_to_log = {"accuracy": accuracy, "logLoss": logLoss, "f1_weighted": f1_weighted, "f1_binary": f1_binary}

    parameters = classifier_fit.get_params()
    log_message(f"the parameters that we got from the default parameters are: {parameters}, the time is: {time.time()-startTime}")
    log_message(f"now we are after the run of the Rendom_forest model with the default parameters and the results on the test data are: {data_to_log}, the time is: {time.time()-startTime}")

run_Rendom_forest()
log_message(f"finished running the Rendom_forest model,the time is: {time.time()-startTime}")
#---------------------------- Rendom_forest with GridSearchCV -------------------------------
def run_Rendom_forest_with_GridSearchCV():
    rf_GS_CV = Rendom_forest_classification_BC_useingGridSearchCV(X_train_np, y_train, X_test_np, y_test)
    best_rf_classifier ,classifier_fit_GS_CV , parameters = rf_GS_CV.build_RandomForestClassifierWithGridSearchCV()
    log_message(f"the parameters that we got from the GridSearchCV are: {parameters}, the time is: {time.time()-startTime}")
    predictions_On_TrainDS = rf_GS_CV.predict_RandomForestClassifierTrainData(classifier_fit_GS_CV)
    predictions_On_TestDS = rf_GS_CV.predict_RandomForestClassifierTestData(classifier_fit_GS_CV)
    #check the accuracy base on accuracy_score log_loss and f1_score
    accuracy , logLoss, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TrainDS, y_train)
    print ("accuracy: ", accuracy
        , "logLoss: ", logLoss
        , "f1_weighted: ", f1_weighted
        , "f1_binary: ", f1_binary)
    data_to_log = {"accuracy": accuracy, "logLoss": logLoss, "f1_weighted": f1_weighted, "f1_binary": f1_binary}
    log_message(f"now we are after the run of the Rendom_forest model with GridSearchCV and the results on the train data are: {data_to_log}, the time is: {time.time()-startTime}")

    accuracy , logLoss, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TestDS, y_test)
    print ("accuracy: ", accuracy
        , "logLoss: ", logLoss
        , "f1_weighted: ", f1_weighted
        , "f1_binary: ", f1_binary)
    data_to_log = {"accuracy": accuracy, "logLoss": logLoss, "f1_weighted": f1_weighted, "f1_binary": f1_binary}
    log_message(f"now we are after the run of the Rendom_forest model with GridSearchCV and the results on the test data are: {data_to_log}, the time is: {time.time()-startTime}")

    log_message(f"finished running the Rendom_forest model with GridSearchCV,the time is: {time.time()-startTime}")
    return best_rf_classifier ,classifier_fit_GS_CV , parameters

run_Rendom_forest_with_GridSearchCV()
