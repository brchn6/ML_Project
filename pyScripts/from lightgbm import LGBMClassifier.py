#%%
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from RunPipe import *
from imblearn.pipeline import make_pipeline as make_impipe
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

lgbm_clf = LGBMClassifier(random_state=41)
xgb_clf = XGBClassifier(random_state=42)
lr = LinearRegression()

scorers = ['roc_auc', 'f1', 'recall', 'neg_log_loss', 'precision', 'accuracy']

lgbm_clf.fit(X_train_np, y_train)

# Assuming you have X_test and y_test as your test data
y_pred = lgbm_clf.predict_proba(X_test_np)[:, 1]
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
roc_auc = accuracy_score(y_test, y_pred)


#%%
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#print("ROC AUC Score:", roc_auc)

def make_cv_pipe(classifier, smote=None):
    if smote:
        cv_pipe = make_impipe(smote, col_processor, classifier)
    else:
        cv_pipe = make_impipe(col_processor, classifier)
    return cv_pipe

cv_pipe = make_cv_pipe(lgbm_clf)

vatcv = cross_validate(cv_pipe, X_train, y_train, cv=cv, scoring=scorers)