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
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, log_loss
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as imbipipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from preprocessing_pipe_copy import *

# Define the classifiers to be evaluated, with default parameters
rnd_clf = BalancedRandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)
lgbm_clf = LGBMClassifier(random_state=42)
catboost_clf = CatBoostClassifier(random_state=42)
logistic_reg = LogisticRegression(random_state=42)

classifiers = [rnd_clf, xgb_clf, lgbm_clf, catboost_clf, logistic_reg]

# Define the scoring metrics (perofrmance measure (pm))
scorers = ['roc_auc', 'f1', 'recall', 'neg_log_loss', 'precision', 'accuracy']

# Initialize cross-validation
X_train = processed.copy()
y_train = processed['readmitted'].astype('category')
X_train = X_train.drop(columns='readmitted')

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

###############3Generating Synthetic data with GAN:##############
#Separating majority and minority classes:
maj_class = processed[processed['readmitted'] == 1]
min_class = processed[processed['readmitted'] == 0]

#Creating metadata (done with builtin SDV class), using processed data before column transformation:
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(processed)

metadata.update_column(
    column_name='diabetesMed',
    sdtype='boolean'
)

metadata.update_column(
    column_name='change',
    sdtype='boolean'
)

#Seting CTGAN and fitting data:
ct_gan = CTGANSynthesizer(
    metadata,
    epochs = 200,
    enforce_min_max_values=True,
    )

ct_gan.fit(min_class)

#Generating synthetic samples:
synthetic_samples_ct = ct_gan.sample(len(maj_class)-len(min_class))

#Evaluate model:
quality_report = evaluate_quality(
    real_data=min_class,
    synthetic_data=synthetic_samples_ct,
    metadata=metadata)

diagnostic_report = run_diagnostic(
    real_data=min_class,
    synthetic_data=synthetic_samples_ct,
    metadata=metadata)

ctgan_diag = diagnostic_report.get_score()
ctgan_score = quality_report.get_score()

with open('ctgan_score.txt', 'w') as f:
    f.write(str(ctgan_score) + '\n' + str(ctgan_diag))

#Visualize comparison to the original data:
from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=min_class,
    synthetic_data=synthetic_samples_ct,
    metadata=metadata,
    column_names=['num_medications', 'num_lab_procedures'],
    )

fig.write_image('myfig.png')

#Creating and exporting balanced df:
balanced_train_set = pd.concat([processed, synthetic_samples_ct])

df_synthetic = pd.DataFrame(balanced_train_set, columns=processed.columns)

df_synthetic.to_csv('balanced_train_set.csv', index=False)

#Use balanced df to test model:
balanced_train_set = pd.read_csv('balanced_train_set.csv')

#Transform balanced df:
y_train_bal = balanced_train_set['readmitted'].copy().astype('category')
X_train_bal = balanced_train_set.drop(columns='readmitted')

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


