#%%
from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from sklearn.model_selection import cross_validate, StratifiedKFold

X_train = X_train
y_train = y_train  
X_train_bal = X_train
y_train_bal = y_train

#Call ClassifierEvaluation class:
classifier_evaluation = ClassifierEvaluator(col_processor, classifier=xgb_clf, score='roc_auc')
#%%
#evaluate XGBoost classifier with roc_auc score:
score, suffix = classifier_evaluation.cv_evaluate(X_train=X_train, y_train=y_train,mode = 'normal', splits=3)

score
# %%
classifier_evaluation = ClassifierEvaluator(col_processor, classifiers=classifiers, scorers=scorers)

st = classifier_evaluation.generate_score_table(X_train=X_train, y_train=y_train, X_train_bal=X_train_bal, y_train_bal=y_train_bal, smote=False, normal=True, splits=3)


