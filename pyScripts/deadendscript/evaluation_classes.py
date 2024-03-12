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
from sklearn.model_selection import cross_validate, StratifiedKFold
from imblearn.pipeline import make_pipeline as make_impipe
from imblearn.over_sampling import SMOTENC
import pandas as pd
import os

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Define the classifiers to be evaluated, with default parameters
rnd_clf = BalancedRandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)
lgbm_clf = LGBMClassifier(random_state=42)
catboost_clf = CatBoostClassifier(random_state=42)
logistic_reg = LogisticRegression(random_state=42)

classifiers = [rnd_clf, xgb_clf, logistic_reg]

# Define the scoring metrics (perofrmance measure (pm))
scorers = ['roc_auc', 'f1', 'recall', 'neg_log_loss', 'precision', 'accuracy']

bool_cols = ['change', 'diabetesMed']

#Cant import from DefPipeLineClasses because of circular import, for now its like this:
cat_cols = ['race', 'gender', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'num_procedures', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'diag_1', 'diag_2',
       'diag_3', 'number_diagnoses', 'A1Cresult', 'metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
       'rosiglitazone', 'acarbose', 'miglitol', 'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-pioglitazone'] + bool_cols

smote = SMOTENC(random_state=42, categorical_features=cat_cols)

class ClassifierEvaluator:
    def __init__(self, col_processor, classifier=None, score=None, scorers = [], classifiers = [], splits=10):
        self.classifier = classifier
        self.scorer = score
        self.scorers = scorers
        self.classifiers = classifiers
        self.col_processor = col_processor
        self.splits = splits

    def cv_evaluate(self, X_train = None, y_train = None, X_train_bal = None, y_train_bal = None, mode = 'normal', classifier = None, scorer = None , splits=None):
        #Check if either one of classifier or scorer is None:
        if (self.classifier is None) | (self.scorer is None):
            if (self.classifiers is None) | (self.scorers is None):
                raise ValueError('Enter classifier and score')
        
        #Check if X_train and y_train are None:
        if (X_train is None) | (y_train is None):
        #If both are None, check if X_train_bal and y_train_bal are None:
            if (X_train_bal is None) | (y_train_bal is None):
                raise ValueError('Enter training data')
        
        #Set default classifier if None:
        if classifier is None:
            classifier = self.classifier
            
         # Set default scorer if None
        if scorer is None:
            scorer = self.scorer

        # Set default splits if None
        if splits is None:
            splits = self.splits

        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        smote = SMOTENC(random_state=42, categorical_features=cat_cols)
        cv_pipe = make_impipe(self.col_processor, classifier)

        if  (mode == 'normal') | (mode == 'smote'):
            X_train_data = X_train
            y_train_data = y_train 
            if mode == 'smote':
                cv_pipe = make_impipe(smote, self.col_processor, classifier)
                suffix = '_sm'
            else:
                suffix = ''
            
        elif mode == 'balanced':
            X_train_data = X_train_bal
            y_train_data = y_train_bal
            suffix = '_gan'

        cross_val_scores = cross_validate(cv_pipe, X_train_data, y_train_data, cv=cv, scoring=scorer)
        if type(scorer) == list:
            return cross_val_scores, suffix 
        else:
            return cross_val_scores['test_score'].mean(), suffix

    def generate_score_table(self, X_train = None, y_train = None, X_train_bal = None, y_train_bal = None, smote=False, balanced=False, normal=False, splits=None):
        #Check if either classifiers or scorers are type list:
        if not (isinstance(self.classifiers, list) and isinstance(self.scorers, list)):
            raise ValueError('Classifiers and scorers must be lists.')
        
        if splits is None:
            splits = self.splits
        
        score_table = pd.DataFrame()
    
        def process_classifier(classifier, mode):
            classifier_name = type(classifier).__name__
            score_dict, suffix = self.cv_evaluate(X_train, y_train, mode=mode, classifier=classifier, scorer=self.scorers, splits=splits)
            for k, v in score_dict.items():
                if k.startswith('test'):
                    score_table.loc[classifier_name + suffix, k[5:]] = v.mean()

        if normal:
            for classifier in self.classifiers:
                process_classifier(classifier, mode='normal')

        if balanced and X_train_bal is not None and y_train_bal is not None:
            for classifier in self.classifiers:
                process_classifier(classifier, mode='balanced')

        if smote:
            for classifier in self.classifiers:
                process_classifier(classifier, mode='smote')

        return score_table , splits


    
