
import os
from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from sklearn.model_selection import cross_validate, StratifiedKFold
from classes.CopulaGenerator import *
from DefPipeLineClasses import *
import pandas as pd

"""
train_set = train_set
The purpose of this file is to generate synthetic data using CopulaGAN and CTGAN, combine each of them to the original dataset
and evaluate the performance of the classifiers using the synthetic data.
Classsifiers evaluated: XGBoost, RandomForest, LogisticRegression, CatBoost, LightGBM
Scores obtained for each classifier: neg_log_los and auc_roc
A table containing the scores for each classifier is generated and saved as a csv file.
Comparisons are made between the original dataset, SMOTE, CopulaGAN and CTGAN dataframes helped us to decide which is will dataframe
will be used for training.
"""
#Started by gnerating synthetic data using CopulaGAN and CTGAN:
#After generating the synthetic data, the data was combined with the original dataset.
#The classes were constructed to ensure rubostness and reusability of the code.

#Synthisizer parameters were set to 300 epochs on both CopulaGAN and CTGAN.
#Boolean_columns were specified.
#Attempt was made to generate synthetic data with 100 epochs and results did not improve.

copula_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                               majority_class_label=1, gans = 'copula',
                                               boolean_columns=['change', 'diabetesMed'], 
                                               epochs=100)

ct_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                            majority_class_label=1, gans = 'ctgan',
                                            boolean_columns=['change', 'diabetesMed'], 
                                            epochs=100)

for gans, name in zip([copula_gans, ct_gans], ['copula', 'ctgan']):
    
    preprocessing = make_pipeline(
        DropColumns(columns_to_drop),
        IDSTransformer(),
        CustomTransformer(bool_functions),
        ConditionalTransformer(condition=True, transformer=gans),
        DiseaseConverter(),
        A1CTransformer(),
        CustomTransformer(functions),
    )
    
    df_transformed = preprocessing.fit_transform(train_set)
    df_name = name + '_train_set_100_epochs'
    
    path = os.path.join(os.getcwd(), '..', 'data')
    csv_filename = df_name + '.csv'
    df_transformed.to_csv(os.path.join(path, csv_filename), index=False)


#This part was done to evaluate the performance of the classifiers using the different datasets.
#The classifiers were evaluated using the original dataset, SMOTE, CopulaGAN and CTGAN datasets.
#The scores obtained for each classifier were saved as a csv file.

#Import original DF for normal and smote evaluation:
#dim - 30134 rows × 38 columns
X_train = X_train
y_train = y_train

#Import synthetic data for copula evaluation:
#dim - 54287 rows × 38 columns
X_train_cop = pd.read_csv(os.path.join(os.getcwd(), '../data','copula_train_set_100_epochs.csv'))
y_train_cop = X_train_cop['readmitted']
X_train_cop = X_train_cop.drop(columns=['readmitted'])

#import synthetic data for ctgan evaluation:
#dim -54415 rows × 38 columns
X_train_ct = pd.read_csv(os.path.join(os.getcwd(), '../data','ctgan_train_set_100_epochs.csv'))
y_train_ct = X_train_ct['readmitted']
X_train_ct = X_train_ct.drop(columns=['readmitted'])

#Call ClassifierEvaluation class:
classifier_evaluation = ClassifierEvaluator(col_processor=col_processor, classifiers=classifiers, scorers=scorers)

#evaluate XGBoost classifier with roc_auc score:
score_table_all = classifier_evaluation.generate_score_table(X_train=X_train, y_train=y_train,
                                                             X_train_cop=X_train_cop, y_train_cop=y_train_cop,
                                                             X_train_ct=X_train_ct, y_train_ct=y_train_ct,
                                                             normal=True, smote=True, bal_cop=True, bal_ct=True,
                                                             splits=10)

csv_filename = 'score_table_all_100_epochs' + '.csv'
path = os.path.join(os.getcwd(), '..', 'data')
score_table_all.to_csv(os.path.join(path, csv_filename), index=True)

#This section will test the performance of the copulGANS which generated with 4 numeric features:
#It will be compared to the other Datasets which were generated in the above section:

#Import synthetic data for copula evaluation:
#dim - (54428, 38)
X_train_cop = pd.read_csv(os.path.join(os.getcwd(), '../data','copula_train_set_300_epochs_4_numeric.csv'))
y_train_cop = X_train_cop['readmitted']
X_train_cop = X_train_cop.drop(columns=['readmitted'])

#Call ClassifierEvaluation class:
classifier_evaluation = ClassifierEvaluator(col_processor=col_processor, classifiers=classifiers, scorers=scorers)

#evaluate XGBoost classifier with roc_auc score:
score_table_all = classifier_evaluation.generate_score_table(X_train_cop=X_train_cop, y_train_cop=y_train_cop,
                                                             normal=False, smote=False, bal_cop=True, bal_ct=False,
                                                             splits=10)

csv_filename = 'score_table_all_100_epochs' + '.csv'
path = os.path.join(os.getcwd(), '..', 'data')
score_table_all.to_csv(os.path.join(path, csv_filename), index=True)

# Dataset with best performance was balanced with CopulaGANsynthesizer