#%%
"""
Main file for the BarModels directory
"""
import warnings

warnings.filterwarnings('ignore')
%reload_ext autoreload
%autoreload 2

#---------------------------- Imports -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  

#---------------------------- Paths -------------------------------
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# ---------------------------- data incoming -------------------------------
#load the train ds
X_train_np = np.load("X_train_np.npy", allow_pickle=True).item()
y_train = np.load("./y_train.npy")
#load the test ds
X_test_np = np.load("X_test_np.npy", allow_pickle=True).item()
y_test = np.load("./y_test.npy")

# instantiate the classifier 
rfc = RandomForestClassifier(random_state=42, n_estimators=10)
# fit the model
rfc.fit(X_train_np, y_train)
# Predict the Test set results
y_pred = rfc.predict(X_train_np)

#%%
accuracy_score(y_test, y_pred)



#%%
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


time_start = time.time()
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
# svm_clf = SVC(gamma="scale", random_state=42)

# voting_clf = VotingClassifier(
#     estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#     voting='hard')
voting_clf = VotingClassifier(
    estimators=[('rf', rnd_clf)],
    voting='hard')

voting_clf.fit(X_train_np, y_train)

y_pred = voting_clf.predict(X_test_np)

time_end = time.time()
print('Time to train the model: {0:0.2f} seconds'.format(time_end - time_start))
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#%%

#load the data:
X_train = pd.read_csv(os.path.join(path, 'X_train_df.csv'))
y_train = pd.read_csv(os.path.join(path, 'y_train.csv'))

X_test = pd.read_csv(os.path.join(path, 'X_test_df.csv'))
y_test = pd.read_csv(os.path.join(path, 'y_test.csv'))

# Compare the columns of X_train and X_test
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
extra_cols = train_cols - test_cols

# Remove any identified extra columns from X_train
if extra_cols:
    X_train = X_train.drop(columns=list(extra_cols))

X_train_np=X_train