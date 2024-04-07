# """
# Main file for the BarModels directory
# """
# Create a logger
import logging
import time
startTime = time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/logs/main.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("now im here starting to import the libraries", "the time is: ", time.time()-startTime)
import warnings
warnings.filterwarnings('ignore')
# %reload_ext autoreload
# %autoreload 2
print("Hello im gonna run")
#---------------------------- Imports -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
root = os.path.dirname(os.path.abspath(__name__))
if root not in sys.path:
    sys.path.append(root)
else:
    pass

here=os.path.dirname(os.path.abspath(__file__))
logger.info("now im here starting to load the data", "the time is: ", time.time()-startTime)


# ---------------------------- data incoming -------------------------------

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
logger.info("now im here starting to run the models", "the time is: ", time.time()-startTime)
# ---------------------------- Rendom_forest -------------------------------
#importing the Rendom_forest_classification_BC_defultParams class
from pyScripts.BarModels.Rendom_forest_BC import Rendom_forest_classification_BC_defultParams
#create a Rendom_forest_classification_BC object
rf = Rendom_forest_classification_BC_defultParams(X_train_np, y_train, X_test_np, y_test)
#build the model
classifier_fit = rf.build_RandomForestClassifier()
#predict the model
predictions_On_TrainDS = rf.predict_RandomForestClassifierTrainData(classifier_fit)
#check the accuracy
accuracy = rf.accuracy_score(predictions_On_TrainDS)
print(accuracy)
print(predictions_On_TrainDS)

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

time_start = time.time()
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(X_train_np, y_train)
y_pred = voting_clf.predict(X_test_np)
time_end = time.time()
logger.info('Time to train the model: {0:0.2f} seconds'.format(time_end - time_start))
logger.info('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#make log file for this cell and store it in the log folder
logging.basicConfig(filename='/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/logs/VotingClassifier.log', level=logging.DEBUG)
logging.info('Time to train the model: {0:0.2f} seconds'.format(time_end - time_start))
logging.info('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
logging.info('run time: {0:0.2f} seconds'.format(time_end - time_start))
#---------------------------- Rendom_forest with GridSearchCV -------------------------------
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_np, y_train)
print(grid_search.best_params_)

