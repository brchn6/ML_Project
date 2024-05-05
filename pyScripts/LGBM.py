# LGBM
import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
import os

# Get the current working directory ## Modify accordingly
# If working on the cloud, load files as follows:
current_directory = os.path.dirname(os.path.abspath('LGBM.py'))
csv_file_path_train = os.path.join(current_directory, "X_train_df.csv")
csv_file_path_train_y = os.path.join(current_directory, "y_train.csv")
csv_file_path_x_test = os.path.join(current_directory, "X_test_df.csv")
csv_file_path_y_test = os.path.join(current_directory, "y_test.csv")
csv_file_path_diabetic_data = s.path.join(current_directory, "diabetic_data.csv")
X_train = pd.read_csv(csv_file_path_train)
X_train = X_train.drop(columns=['diag_3_365.44','repaglinide_Down'])
y_train = pd.read_csv(csv_file_path_train_y)
X_test = pd.read_csv(csv_file_path_x_test)
y_test = pd.read_csv(csv_file_path_y_test)
original_train_set = pd.read_csv(csv_file_path_diabetic_data)

# if working locally, load files as follows:
# os.chdir('C:\Git\ML_project\pyScripts\GuyTrain')
# X_train = pd.read_csv('X_train_df.csv')
# X_train = X_train.drop(columns=['diag_3_365.44','repaglinide_Down'])
# y_train = pd.read_csv('y_train.csv')
# X_test = pd.read_csv('X_test_df.csv')
# y_test = pd.read_csv('y_test.csv')
# original_train_set = pd.read_csv('diabetic_data.csv')

train_dataset = lgbm.Dataset(X_train, label=y_train)

# # divide the train set to train and validation in order to fix the number of trees.
X_train_es, X_val, y_train_es, y_val = train_test_split(X_train, y_train, shuffle=True,random_state=42)

train_d = lgbm.Dataset(X_train_es, label=y_train_es)
valid_d = lgbm.Dataset(X_val, label=y_val)
lr_range = [0.0001,0.001,0.01,0.1,0.2,0.5]
for lr in lr_range:
    params_es = {'objective': 'binary','learning_rate': lr}
    lgbm_es = lgbm.train(params=params_es,train_set=train_d, valid_sets=[valid_d], callbacks=[lgbm.early_stopping(stopping_rounds=10)])
    num_estimators = lgbm_es.best_iteration
    params_es['n_estimators']=num_estimators
    print('learning_rate: ',lr)
    print('best_iteration: ',lgbm_es.best_iteration)
    
# # Best learning rate is 0.1, it dictates n_estimators = 95. with a valid_0 loss = 0.200864.

# # Evaluate untuned model on the train set:
params_train = {'objective': 'binary',
        'learning_rate': 0.1,  
        'num_boost_round' : 95
}

train_all = lgbm.Dataset(X_train, label=y_train)
lgbm_train_p = lgbm.train(params=params_train,train_set=train_all)
y_pred_train_prob = lgbm_train_p.predict(X_train)
neg_log_loss_train = -log_loss(y_train, y_pred_train_prob)
print("Negative log loss on training set:", neg_log_loss_train)
# # Negative log loss on training set: -0.17860238301838519. with params_train

# # ## Base CV score, after fixing n_estimators = 95.

base_cv = lgbm.cv(params=params_train,train_set=train_dataset,nfold=10,metrics='binary_logloss',num_boost_round=95)
cv_logloss = cross_val_score(estimator, X_train, y_train, cv=10, scoring='neg_log_loss')
mean_logloss = np.mean(cv_logloss)
std_logloss = np.std(cv_logloss)
print("Mean neg log loss:", mean_logloss)
print("STD neg log loss:", std_logloss)

log_losses = base_cv['binary_logloss-mean']
print('Mean Log Loss:',np.mean(log_losses))
print('Standard Deviation Log Loss:',np.std(log_losses))

# ## Hyper-Parameters tuning:

params = {'objective': 'binary',
        'learning_rate': 0.1,
        'n_estimators': 95}

params_grid_a = {
                'boosting_type':['gbdt','dart','rf'],
                'max_depth': range(-1,20,1),
                'num_leaves': [5,20,31, 80, 320]
            }

params_grid_b = {
    'subsample':np.arange(0,1.2,0.2),
    'colsample_bytree':np.arange(0,1.2,0.2),
    'subsample_freq': [0,1],
    'min_child_samples': range(0,100,20)
}

params_grid_c = {   
    'reg_alpha': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10],
    'reg_lambda': [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10] 
                }

params_grid_d = {'learning_rate':np.arange(0,1,0.001),
                'is_unbalance':['True','False'],
                'scale_pos_weight': np.arange(1,10,1)
                }

# # ## Hyper-parameter tuning using GridSearchCV

grids = [params_grid_a,params_grid_b,params_grid_c,params_grid_d]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for grid in grids:
    estimator = lgbm.LGBMClassifier(random_state=42,**params)
    gsearch = GridSearchCV(param_grid=grid, estimator=estimator,
    scoring='neg_log_loss', cv=cv)
    gsearch.fit(X_train,y_train)
    print(gsearch.best_params_)
    print('neg_log_loss', gsearch.best_score_)
    params.update(gsearch.best_params_)
print(params)

# # # Create a RandomizedSearchCV Object
estimator = lgbm.LGBMClassifier(random_state=42,**params)
params_grid_random = {
    'min_split_gain':np.arange(0,1),
    'min_gain_to_split':np.arange(0,1),
    'min_child_weight': np.arange(0,100),
    'max_bin': np.arange(10, 1000),
    'min_sum_hessian_in_leaf': np.arange(0,10)}

random_search = RandomizedSearchCV(estimator=estimator, param_distributions=params_grid_random, n_iter=50, scoring='neg_log_loss', cv=cv,random_state=42)

# # # # Fit RandomizedSearchCV Object
random_search.fit(X_train, y_train)

print("Best Parameters found: ", random_search.best_params_)
print("Best Score found: ", random_search.best_score_)
params.update(random_search.best_params_)
print("Params:",params)

## Predict the test-set with the tuned model, best parameters
params_tuned = {
                'verbosity':0,
                'objective': 'binary',
                'learning_rate': 0.146,
                'n_estimators': 95,
                'boosting_type': 'gbdt',
                'max_depth': 12,
                'num_leaves': 31,
                'colsample_bytree': 1.0,
                'min_child_samples': 80,
                'subsample': 0.8,
                'subsample_freq': 1,
                'reg_alpha': 1e-07,
                'reg_lambda': 1e-08,
                'is_unbalance': 'True',
                'scale_pos_weight': 1,
                'min_sum_hessian_in_leaf': 0,
                'min_split_gain': 0,
                'min_gain_to_split': 0,
                'min_child_weight': 18,
                'max_bin': 742,
                'importance_type':'gain'}

estimator = lgbm.LGBMClassifier(random_state=42,**params_tuned)

# Predict over 15 random states:
from sklearn.metrics import log_loss, precision_score, recall_score,roc_auc_score, accuracy_score

def calculate_metrics_and_importance(X_train, X_test, y_train, y_test):
    model = estimator
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    neg_logloss = log_loss(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)

    feature_importance = model.feature_importances_
    feature_names = model.feature_name_

    return precision, recall, neg_logloss, roc_auc, accuracy, feature_importance, feature_names

# Initialize lists to store metrics and feature importance for each state
precisions = []
recalls = []
neg_log_losses = []
roc_aucs = []
feature_importances = []
accuracy_list = []
all_feature_names = None

# Loop over 15 random states
for state in range(15):
    estimator = lgbm.LGBMClassifier(random_state=state, **params_tuned)
    precision, recall, neg_log_loss, roc_auc,accuracy, feature_importance, feature_names = calculate_metrics_and_importance(X_train, X_test, y_train, y_test)

    precisions.append(precision)
    recalls.append(recall)
    neg_log_losses.append(neg_log_loss)
    roc_aucs.append(roc_auc)
    feature_importances.append(feature_importance)
    accuracy_list.append(accuracy)
    if all_feature_names is None:
        all_feature_names = feature_names

# Create dataframe with results
results_df = pd.DataFrame({
    'Precision': precisions,
    'Recall': recalls,
    'Neg Log Loss': neg_log_losses,
    'ROC AUC': roc_aucs,
    'Accuracy': accuracy
})

##################
def get_feature_name(strings, exceptional_strings):
    modified_strings = []
    for string in strings:
        # Remove '_Uncontrolled' part if it exists
        string = string.replace('_Uncontrolled', '')
        
        if string not in exceptional_strings:
            parts = string.split('_')
            if len(parts) > 1:
                modified_string = '_'.join(parts[:-1])
                modified_strings.append(modified_string)
            else:
                modified_strings.append(string)
        else:
            modified_strings.append(string)
    return modified_strings

numerical_features = original_train_set.select_dtypes(include='number')
numerical_features_cols = numerical_features.columns
feature_names_transformed = get_feature_name(all_feature_names,numerical_features_cols)
print(feature_names_transformed)
feature_importance_df = pd.DataFrame(feature_importances)
feature_importance_df_scaled = pd.DataFrame(feature_importances / np.sum(feature_importances))

def featureImportancePlot (feature_names, table):
    
    table_mean = table.mean(axis = 0)
    table_std = table.std(axis = 0)

    #Transform the feature names:
    feature_names = get_feature_name(feature_names,numerical_features_cols)

    #Change first column names:
    feature_names[0] = 'number_emergency'
    feature_names[1] = 'number_outpatient'
    
    feature_count = pd.DataFrame(feature_names).value_counts()
    
    sums = {}
    stds = {}
    for i in range(len(feature_names)):
        name = feature_names[i]
        if name in sums:
            sums[name] += table_mean[i]
            stds[name] += table_std[i]
        else:
            sums[name] = table_mean[i]
            stds[name] = table_std[i]
    
    #Divide the stds by the feature count:
    for key in stds:
        stds[key] = stds[key]/feature_count[key]
    
    #sort the sums in descending order:
    sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))
    
    #order stds by sums:
    stds = {k: stds[k] for k in sums.keys()}
    
    #plot the feature importances according to the sums dictionary (which is sorted):
    plt.figure(figsize=(10, 5))
    
    error = list(stds.values())[:10][::-1]
    plt.barh(list(sums.keys())[:10][::-1], list(sums.values())[:10][::-1], align='center', xerr=error, color='skyblue', alpha=0.7)
    print(list(sums.keys())[:10])
    # Add labels and title
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.title('Top 10 Feature Importance on (mean of 15 seeds)')
    plt.tight_layout()
    
    #export the feature the plot:z
    plt.savefig('feature_importance_15_seeds_mean.png')
    
    plt.show()
featureImportancePlot(all_feature_names,feature_importance_df)
feature_importance_df.columns = all_feature_names

# # Save results dataframes to CSV
results_df.to_csv('results_lgbm.csv', index=False)
feature_importance_df.to_csv('results_importance_lgbm.csv', index=False)

