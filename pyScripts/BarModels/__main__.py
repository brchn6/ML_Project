# """
# Main file for the BarModels directory
# """
# Create a logger
#---------------------------- Imports basic -------------------------------
import sys
import os
import logging
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
startTime = time.time()
root = os.path.dirname(os.path.abspath(__name__))
if root not in sys.path:
    sys.path.append(root)
else:
    pass
here=os.path.dirname(os.path.abspath(__file__))

# %reload_ext autoreload
# %autoreload 2
#---------------------------- Imports for the model -------------------------------
from pyScripts.BarModels.Rendom_forest_BC import log_message
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_defultParams
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_useingGridSearchCV
# ---------------------------- data incoming -------------------------------
log_message(f"starting to load the data, the time is: {time.time()-startTime}")
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
log_message(f"finished loading the data, the time is: {time.time()-startTime}")

# ---------------------------- Rendom_forest -------------------------------
log_message(f"starting to run the Rendom_forest model,the time is: {time.time()-startTime}")
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


accuracy , logLoss, f1_weighted, f1_binary = rf.accuracy_score(predictions_On_TestDS, y_test)
print ("accuracy: ", accuracy
       , "logLoss: ", logLoss
       , "f1_weighted: ", f1_weighted
       , "f1_binary: ", f1_binary)

log_message(f"finished running the Rendom_forest model,the time is: {time.time()-startTime}")
#---------------------------- Rendom_forest with GridSearchCV -------------------------------
rf_GS_CV = Rendom_forest_classification_BC_useingGridSearchCV(X_train_np, y_train, X_test_np, y_test)
best_rf_classifier ,classifier_fit_GS_CV = rf_GS_CV.build_RandomForestClassifierWithGridSearchCV()
predictions_On_TrainDS = rf_GS_CV.predict_RandomForestClassifierTrainData(classifier_fit_GS_CV)
predictions_On_TestDS = rf_GS_CV.predict_RandomForestClassifierTestData(classifier_fit_GS_CV)
#check the accuracy base on accuracy_score log_loss and f1_score
accuracy , logLoss, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TrainDS, y_train)
print ("accuracy: ", accuracy
       , "logLoss: ", logLoss
       , "f1_weighted: ", f1_weighted
       , "f1_binary: ", f1_binary)

accuracy , logLoss, f1_weighted, f1_binary = rf_GS_CV.accuracy_score(predictions_On_TestDS, y_test)
print ("accuracy: ", accuracy
       , "logLoss: ", logLoss
       , "f1_weighted: ", f1_weighted
       , "f1_binary: ", f1_binary)
log_message(f"finished running the Rendom_forest model with GridSearchCV,the time is: {time.time()-startTime}")



