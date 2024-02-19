#%%
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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# Import variables from Run_pipeline module
from Run_pipeline import *
from imblearn.pipeline import Pipeline
cat_cols = cat_cols

# Define your data
# Ensure that diabetes_labels and diabetes_test are defined earlier in your code

#%%

class SmoteUp(BaseEstimator, TransformerMixin):

    def __init__(self, labels, cols):
        self.labels = labels
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sm = SMOTENC(random_state=42, categorical_features=self.cols)
        X_res, y_res = sm.fit_resample(X, self.labels)
        return X_res, y_res
#%%
smoted_data = SmoteUp(diabetes_labels, cat_cols)
Xs, yx = smoted_data.transform(diabetes_test)

# %%
