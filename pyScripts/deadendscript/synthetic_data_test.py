"""
description:
1. This script is a deadend script, it is used to generate synthetic data using CTGAN and SMOTE
2. The script is used to compare the performance of the classifiers on the original data and the synthetic data
3. the output of this script is preformance of the classifiers on the original data and the synthetic data
4. another output is the synthetic data together with the original data to a csv file called balanced_train_set.csv

total output:
1. score_table.csv
2. balanced_train_set.
3. ctgan_score.txt
4. myfig.png

Author: Guy Ilan
"""

#-------------------------------------------imports-------------------------------------------
import os
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as imbipipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from RunPipe import *

# Define the classifiers to be evaluated, with default parameters
rnd_clf = BalancedRandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)
lgbm_clf = LGBMClassifier(random_state=42)
catboost_clf = CatBoostClassifier(random_state=42)
logistic_reg = LogisticRegression(random_state=42)

classifiers = [rnd_clf, xgb_clf, lgbm_clf, catboost_clf, logistic_reg]

# Define the scoring metrics (perofrmance measure (pm))
scorers = ['roc_auc', 'f1', 'recall', 'neg_log_loss', 'precision', 'accuracy']

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smote = SMOTENC(random_state=42,categorical_features=cat_cols)

###########Run with and without SMOTE:###########

score_table = pd.DataFrame()

# Create a function to make the pipeline
def make_cv_pipe(classifier, smote=None):
    if smote:
        cv_pipe = make_impipe(smote, col_processor_s, classifier)
    else:
        cv_pipe = make_impipe(col_processor_s, classifier)
    return cv_pipe

for defs in [None, smote]:
    for classifier in classifiers:
        classifier_name = type(classifier).__name__
        cv_pipe = make_cv_pipe(classifier, smote=defs)

        cross_val_scores = cross_validate(cv_pipe, X_train, y_train, cv=cv, scoring=scorers)
        
        for scoring_metric, scores in cross_val_scores.items():
            if scoring_metric not in ['fit_time', 'score_time']:  # Exclude fit_time and score_time
                if scoring_metric.startswith('test_'):
                    scoring_metric = scoring_metric[5:]  # Remove the 'test_' prefix
                if defs is None:
                    score_table.loc[classifier_name, scoring_metric] = scores.mean()
                else:
                    score_table.loc[classifier_name + '_sm', scoring_metric] = scores.mean()


#############Run balanced df through classifiers:###########
for classifier in classifiers:
        classifier_name = type(classifier).__name__
        cv_pipe = make_cv_pipe(classifier)
        cross_val_scores = cross_validate(cv_pipe, X_train_bal, y_train_bal, cv=cv, scoring=scorers)
        
        for scoring_metric, scores in cross_val_scores.items():
            if scoring_metric not in ['fit_time', 'score_time']:  # Exclude fit_time and score_time
                if scoring_metric.startswith('test_'):
                    scoring_metric = scoring_metric[5:]  # Remove the 'test_' prefix
                    score_table.loc[classifier_name+'_gan', scoring_metric] = scores.mean()    

#Export finished table:
score_table.to_csv('score_table.csv')

#%%

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
import pandas as pd

class ClassifierEvaluator:
    def __init__(self, classifiers, scorers):
        self.classifiers = classifiers
        self.scorers = scorers
    
    def evaluate(self, X_train, y_train, balanced=False, smote=None):
        score_table = pd.DataFrame()
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        smote = SMOTENC(random_state=42, categorical_features=cat_cols)
        
        for classifier in self.classifiers:
            classifier_name = type(classifier).__name__
            cv_pipe = self.make_impipe(col_processor, classifier)
            
            if balanced:
                X_train_data = X_train
                y_train_data = y_train

            else:
                X_train_data = X_train
                y_train_data = y_train
                
                if not smote:  # Apply SMOTE if data is unbalanced
                    X_train_data, y_train_data = smote.fit_resample(X_train_data, y_train_data)

            cross_val_scores = cross_validate(cv_pipe, X_train_data, y_train_data, cv=cv, scoring=self.scorers)

            for scoring_metric, scores in cross_val_scores.items():
                if scoring_metric not in ['fit_time', 'score_time']:  # Exclude fit_time and score_time
                    if scoring_metric.startswith('test_'):
                        scoring_metric = scoring_metric[5:]  # Remove the 'test_' prefix
                    if balanced:
                        score_table.loc[classifier_name, scoring_metric] = scores.mean()
                    else:
                        score_table.loc[classifier_name + '_sm', scoring_metric] = scores.mean()

        return score_table

# Usage example:
# Define classifiers and scorers as before
classifier_evaluator = ClassifierEvaluator(classifiers, scorers)
score_table = classifier_evaluator.evaluate(X_train, y_train)
score_table.to_csv('score_table.csv')