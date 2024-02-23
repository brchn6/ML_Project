import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
import os

os.chdir('C:\ML_diabetes_Saar')
data = pd.read_csv(r'train_set_ctgan.csv')


## make sure you ran script number two that includes the new function to_categorical that converts
## dtype object to categorical. otherwise you will get an error.

# the order of the pipeline should be as describe below:
# step 1: smote, step 2: pre-processing, step 3: the classifier with default parameters.
# the classifiers should be defined before as classes.

classifiers = ['XGboost','LGBM','Catboost','SVM','BalancedRandomForestClassifier']
score = ['neg_log_loss','accuracy','precision','recall','f1','roc_auc']
sm = SMOTENC(random_state=42,categorical_features="auto")
for c in classifiers:
    for s in score:
        pipeline = imbpipeline(steps = [['smote', sm,
                                        ['here insert our pre-preocessing pipe-line'],
                                        ['here insert the classifier of choice, or the iterator (c) 
                                        'if you are looping through a list of iterators']])

        cv = stratified_kfold = StratifiedKFold(n_splits=10,
                                            shuffle=True,
                                            random_state=42)
        cross_val_score = cross_val_score(pipeline, X_train,y_train,cv=cv,scoring = s).mean()
    
        # you will need to add here a step that stores the cross val scores for each iteration.


