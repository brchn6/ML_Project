# ------------------------------ code ------------------------------ #
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as imbpipeline

# Import variables from Run_pipeline module
from Run_pipeline import *
from imblearn.pipeline import Pipeline
cat_cols = cat_cols

# Define your data
# Ensure that diabetes_labels and diabetes_test are defined earlier in your code
y_train = diabetes_labels
X_train = diabetes_test

# Define the classifiers to be evaluated
classifiers = [XGBClassifier, LGBMClassifier, CatBoostClassifier, SVC, BalancedRandomForestClassifier]

# Define the scoring metrics
score = ['neg_log_loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Initialize SMOTENC for handling categorical features
sm = SMOTENC(random_state=42, categorical_features=cat_cols)

# Initialize cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create an empty dictionary to store cross-validation scores
cv_scores = {}

# Iterate over classifiers
for classifier in classifiers:
    # Iterate over scoring metrics
    for s in score:
        # Define the pipeline
        pipeline = Pipeline(steps=[
            ['smote', sm],
            ["classifier", classifier()]
        ])

        # Perform cross-validation
        cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=s)

        # Store cross-validation scores
        classifier_name = classifier.__name__  # Get the name of the classifier
        if classifier_name not in cv_scores:
            cv_scores[classifier_name] = {}
        cv_scores[classifier_name][s] = cross_val_scores

# Now, cv_scores dictionary contains cross-validation scores for each classifier and each scoring metric
# You can analyze or further process these scores as needed
        
#data output as tbl
import pandas as pd
df = pd.DataFrame(cv_scores)
df.to_csv('cv_scores.csv')
# ------------------------------ end ------------------------------ #