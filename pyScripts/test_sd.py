#%%
from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from classes.CopulaGenerator import *
from DefPipeLineClasses import *
import pandas as pd

train_set = train_set

copula_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                               majority_class_label=1, 
                                               gans = 'ctgan',
                                               boolean_columns=['change', 'diabetesMed'], 
                                               epochs=300)

ct_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                            majority_class_label=1, gans = 'ctgan',
                                            boolean_columns=['change', 'diabetesMed'], 
                                            epochs=300)

for gans, name in [copula_gans, ct_gans], ['copula', 'ctgan']:
    preprocessing = make_pipeline(
        DropColumns(columns_to_drop),
        IDSTransformer(),
        CustomTransformer(bool_functions),
        ConditionalTransformer(condition=True, transformer=gans),
        DiseaseConverter(),
        A1CTransformer(),
        CustomTransformer(functions),
    )
    
    df_name = name + '_train_set'
    df_name = preprocessing.fit_transform(train_set)
    df = convertIdsColumn(df_name)
    path = os.path.join(os.getcwd(), '..', 'data')
    df_name.to_csv(os.path.join(path,df_name+'.csv'), index=False)

#Call ClassifierEvaluation class:
#classifier_evaluation = ClassifierEvaluator(col_processor, classifier=xgb_clf, score='neg_log_loss')

#evaluate XGBoost classifier with roc_auc score:
#score, suffix = classifier_evaluation.cv_evaluate(X_train_bal=X_train_bal, y_train_bal=y_train_bal, mode = 'balanced', splits=3)
#%%
classifier_evaluation = ClassifierEvaluator(col_processor, classifiers=classifiers, scorers=scorers)

st = classifier_evaluation.generate_score_table(X_train=X_train, y_train=y_train,
                                                X_train_cop=X_train_cop, y_train_cop=y_train_cop, 
                                                X_train_ct=X_train_ct, y_train_ct=y_train_ct, 
                                                balanced=True ,smote=False, normal=True, splits=3)

boolean_columns = ['diabetesMed', 'change']

train_data = preprocessing.fit_transform(train_set)

generator = CopulaGANSyntheticDataGenerator(label_column='readmitted', 
                                            minority_class_label=0, majority_class_label=1, 
                                            gans='ctgan', boolean_columns=boolean_columns, 
                                            epochs=50)

aa = generator.fit_transform(train_data, export = True, name='balanced_train_set_ct')





