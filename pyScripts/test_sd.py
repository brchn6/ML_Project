#%%
from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from classes.CopulaGenerator import *
from DefPipeLineClasses import *

X_train = X_train
y_train = y_train  
X_train_bal = pd.read_csv(os.getcwd() + '/../data/balanced_train_set.csv')
y_train_bal = X_train_bal['readmitted']
X_train_bal = X_train_bal.drop(columns=['readmitted'])

X_train_bal = convertIdsColumn(X_train_bal)

preprocessing = make_pipeline(
    DropDup(dropdup_col),
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
)

#X_train_bal = preprocessing.fit_transform(X_train_bal)

#Call ClassifierEvaluation class:
classifier_evaluation = ClassifierEvaluator(col_processor, classifier=xgb_clf, score='neg_log_loss')
#%%
#evaluate XGBoost classifier with roc_auc score:
score, suffix = classifier_evaluation.cv_evaluate(X_train_bal=X_train_bal, y_train_bal=y_train_bal, mode = 'balanced', splits=3)

score
# %%
classifier_evaluation = ClassifierEvaluator(col_processor, classifiers=classifiers, scorers=scorers)

st = classifier_evaluation.generate_score_table(X_train=X_train, y_train=y_train, X_train_bal=X_train_bal, y_train_bal=y_train_bal, balanced=True ,smote=False, normal=True, splits=3)
#%%
boolean_columns = ['diabetesMed', 'change']
train_data = preprocessing.fit_transform(train_set)

#%%
generator = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0, majority_class_label=1, boolean_columns=boolean_columns, epochs=50, gans='ctgan')
aa = generator.fit_transform(train_data, export = True, name='balanced_train_set_ct')





# %%
