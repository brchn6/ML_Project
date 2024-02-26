
# ------------------------------ code ------------------------------ #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as imbipipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

# Import variables from Run_pipeline module
from preprocessing_pipe import *

# Define the classifiers to be evaluated, with default parameters

rnd_clf = RandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)
lgbm_clf = LGBMClassifier(random_state=42)
catboost_clf = CatBoostClassifier(random_state=42)
logistic_reg = LogisticRegression(random_state=42)

#Makeing classifiers list:
classifiers = [rnd_clf, xgb_clf, lgbm_clf, catboost_clf, logistic_reg]

# Define the scoring metrics (perofrmance measure (pm))
scoring = {
    'neg_log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True),
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr') 
}
# Initialize cross-validation
X_train = processed.copy()
y_train = processed['readmitted'].astype('category')
X_train = X_train.drop(columns='readmitted')

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
smote = SMOTENC(random_state=42,categorical_features=cat_cols)

#Initialize dictionary to store cross-validation scores
cv_scores = {}
smote = SMOTENC(random_state=42,categorical_features=cat_cols)
for classifier in classifiers:
    classifier_name = type(classifier).__name__
    cv_scores[classifier_name] = {}
    cv_pipe = make_impipe(
        smote,
        col_processor_s,                 
        classifier)

    for s in scoring:
        cross_val_scores = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring=scoring[s])
        cv_scores[classifier_name][s] = cross_val_scores
        if classifier_name not in cv_scores:
            cv_scores[classifier_name] = {}
        cv_scores[classifier_name][s] = cross_val_scores
print(cv_scores)

model_scores = {}

for classifier in classifiers:
    classifier_name = type(classifier).__name__
    model_scores[classifier_name] = {}

    # Fit the classifier to the data
    classifier.fit(X_train, y_train)

    # Predict probabilities for log loss and roc_auc
    if hasattr(classifier, "predict_proba"):
        y_pred_proba = classifier.predict_proba(X_train)
    else:
        y_pred_proba = None

    # Calculate scores for each metric
    for s in scoring:
        if s == 'neg_log_loss' or s == 'roc_auc':
            if y_pred_proba is not None:
                score = scoring[s](classifier, X_train, y_train)
            else:
                score = None
        else:
            y_pred = classifier.predict(X_train)
            score_func = scoring[s]
            score = score_func._score_func(y_encoded, y_train)

        model_scores[classifier_name][s] = score

# Now, cv_scores dictionary contains cross-validation scores for each classifier and each scoring metric
# You can analyze or further process these scores as needed

#Creating table with classifiers and mean of the cv scores in each pm
#Need to change names of the pm's in the df..

models_test_table = pd.DataFrame()

for classifier in cv_scores.keys():
    for pm in cv_scores[classifier].keys():
        models_test_table.loc[classifier,(pm+'-cv')] = cv_scores[classifier][pm].mean()

for classifier in model_scores.keys():
    for pm in model_scores[classifier].keys():
        models_test_table.loc[classifier,pm] = model_scores[classifier][pm]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#data output as tbl
import pandas as pd
df = models_test_table
df.to_csv('cv_scores_full.csv', index=True)

#%%
cv_df = pd.read_csv('cv_scores_full.csv').set_index('Unnamed: 0').rename_axis('Classifier')
# ------------------------------ end ------------------------------ #


