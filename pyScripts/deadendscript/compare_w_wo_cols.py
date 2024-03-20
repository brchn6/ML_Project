
#%%
import os
#from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from classes.CopulaGenerator import *
from DefPipeLineClasses import *
import pandas as pd
from sklearn.compose import make_column_selector as selector
#%%
X_train_w = pd.read_csv(os.path.join(os.getcwd(), '../data','copula_train_set_300_epochs_4_numeric.csv'))
y_train_w = X_train_w['readmitted']
X_train_w = X_train_w.drop(columns=['readmitted'])

def convert_to_float64(dataframe, columns):
    dataframe[columns] = dataframe[columns].astype('float64')
    return dataframe

cols_to_change = ['number_emergency', 'number_outpatient']

#Define the transformer for the NEW numeric features:
num_transformer_none = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

X_train_w = convert_to_float64(X_train_w, cols_to_change)

X_train_wo = X_train_w.drop(columns=cols_to_change)

scorers = ['neg_log_loss', 'f1', 'accuracy']

classifiers = [rnd_clf, xgb_clf, lgbm_clf]

for df, name in zip([X_train_w, X_train_wo], ['with', 'without']):

    #Define the transformer for the original numeric features:
    col_processor = make_column_transformer(
        (num_transformer_none, selector(dtype_include="float64")),
        (num_transformer, selector(dtype_include="int64")),
        (bool_transformer, selector(dtype_include="bool")),
        (cat_transformer, selector(dtype_include="object")),
        n_jobs=3
    )

    #Call ClassifierEvaluation class:
    classifier_evaluation = ClassifierEvaluator(col_processor=col_processor, classifiers=classifiers, scorers=scorers)

    #evaluate XGBoost classifier with roc_auc score:
    score_table_all = classifier_evaluation.generate_score_table(X_train_cop=df, y_train_cop=y_train_w,
                                                                 normal=False, smote=False, bal_cop=True, bal_ct=False,
                                                                 splits=10)

    csv_filename = 'score_table_' + name + 'cols.csv'
    path = os.path.join(os.getcwd(), '..', 'data')
    score_table_all.to_csv(os.path.join(path, csv_filename), index=True)

#%%

#call and concatenate the score tables:
score_table_w = pd.read_csv('../data/score_table_withcols.csv')
score_table_wo = pd.read_csv('../data/score_table_withoutcols.csv')

score_table = pd.concat([score_table_w, score_table_wo], axis=0)

score_table.iloc[0:3,0] = score_table.iloc[0:3,0].apply(lambda x: x + '_with' for x in score_table.iloc[0:3,0])
score_table.iloc[3:6,0] = score_table.iloc[3:6,0].apply(lambda x: x + '_without' for x in score_table.iloc[3:6,0])

#export score table to csv:
score_table.to_csv('../data/score_table_w_vs_wo.csv', index=False)


# %%
a = pd.read_csv('../data/score_table_w_vs_wo.csv', index_col=0)
#make index name none:
