from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from deadendscript.synthetic_data_test import *

lgbm_clf = LGBMClassifier(random_state=42)

lgbm_clf.fit(X_train_bal, y_train_bal)

# Assuming you have X_test and y_test as your test data
y_pred = lgbm_clf.predict_proba(X_test_np)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred)

print("ROC AUC Score:", roc_auc)