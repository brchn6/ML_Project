#Data:
'''
The data folder contains all the csv files given to us at the beginning of the project as well as the final train set used for trqaining the model:
1. diabetes_data.csv
2. IDS_mapping.csv
3. copula_train_set_300_epochs_4_numeric.csv

The folder also contains the score_table.csv generated to estimate the performance of different classifiers:

'''

#pyScripts folder:

#classes:
'''
1. ConditionalTransformer.py - This class is used to conditionally apply a transformer to the data. 
If the condition is set to True, the transformer is applied to the data, else the data is returned as is.
This class was used in the pipline in order to make the GANS class conditional.

2. CopulaGenerator.py - This class is used to generate synthetic data in order to balance the dataset.
Both ctGAN and copulaGAN can be used to generate and balance the dataset.
It was used in order to create the final balanced dataset.

3. evaluation_classifiers.py - This class is used to evaluate the performance of different classifiers on different datasets.
The class is used to compare the performance of the classifiers on the original, smote, ctGAN, and copulaGAN datasets.
The class is used to generate the score_table.csv file.
'''

#deadendscript:

'''
1. disease_ids_conds.py - This script contains function used for recategorizing the data in the dataset, 
as well as to store long lists usedto gnerate the dataset.

2. feature_importance_rnd_clf.py - This script is used to generate the feature importance plot of the dataset using the random forest classifier.

3.generate_n_test_df.py - This script is used to generate the balanced datasets and evaluate the performance of the classifiers on the different datasets.
The classes CopulaGenerator, ConditionalTransformer, and ClassifierEvaluator are used in this script.
'''

#GuyTrain:

'''
1. [X_test_df, X_train_df, y_test_df, y_train_df] - the final train and test sets used for training the model, 
customized to fit the xgboost model.

2. xgboost_gridcv.py - This script was used to hyperparameter tune the xgboost model using grid search cross validation.

3. xgboost_optuna.py - This script was used to hyperparameter tune the xgboost model using optuna.

4. feature_importance_script.py - contains the functions used in order to generate
 the feature importance plot of the xgboost model after hyperparameter tuning.


'''


