from RunPipe import *
from deadendscript.synthetic_data_test import *
from DefPipeLineClasses import *

X_train = X_train
y_train = y_train  
X_train_bal = X_train
y_train_bal = y_train

#Call ClassifierEvaluation class:
classifier_evaluation = ClassifierEvaluator(xgb_clf, 'roc_auc', col_processor)

#evaluate XGBoost classifier with roc_auc score:
score, suffix = classifier_evaluation.cv_evaluate(X_train=X_train, y_train=y_train, smote=False, normal=True, splits=3)

score