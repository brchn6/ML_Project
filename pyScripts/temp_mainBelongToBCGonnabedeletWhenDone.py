#%%
#-----------------Import the necessary packages-----------------
# from main import *
import pandas as pd
import numpy as np
from featureImportanceDir.Feature_ImportanceClass import Build_Feature_Importance
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#-----------------load the data so i dont need to run the whole thing-----------------
X_train = pd.read_csv('../data/copula_train_set.csv')
y_train = X_train['readmitted']
X_train = X_train.drop('readmitted', axis=1)
X_train_np = np.load('featureImportanceDir/X_train_np.npy', allow_pickle=True).item()

#get the feature names from the X_train_np
from RunPipe import col_processor
feature_names = col_processor.get_feature_names_out()
#-------------------------------------------------------------------------------------
# Create the model
lgb_model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

# Create the instance of the class
lgb_inst = Build_Feature_Importance(lgb_model, X_train_np, y_train, feature_names)
xgb_inst = Build_Feature_Importance(xgb_model, X_train_np, y_train, feature_names)
rf_inst = Build_Feature_Importance(rf_model, X_train_np, y_train, feature_names)

# Fit the model
lgb_inst.fitModel()
xgb_inst.fitModel()
rf_inst.fitModel()

# Calculate the feature importance
lgb_importance, lgb_features = lgb_inst.calculateFeatureImportance()
xgb_importance, xgb_features = xgb_inst.calculateFeatureImportance()
rf_importance, rf_features = rf_inst.calculateFeatureImportance()


# Plot the feature importance
lgb_inst.plotFeatureImportance()
xgb_inst.plotFeatureImportance()
rf_inst.plotFeatureImportance()

# Calculate the sum of the original features
lgb_sums = lgb_inst.calculateSumOfOriginalFeatures()
xgb_sums = xgb_inst.calculateSumOfOriginalFeatures()
rf_sums = rf_inst.calculateSumOfOriginalFeatures()


# Plot the sum of original features
lgb_inst.plotOriginalFeatureSums()
xgb_inst.plotOriginalFeatureSums()
rf_inst.plotOriginalFeatureSums()

# Save the feature importance and feature sums
# save all figures in dir names figures
import os
if not os.path.exists('featureImportanceDir/figures'):
    os.makedirs('figures')
lgb_inst.plotFeatureImportance('featureImportanceDir/figures/lgb_feature_importance.png')
xgb_inst.plotFeatureImportance('featureImportanceDir/figures/xgb_feature_importance.png')
rf_inst.plotFeatureImportance('featureImportanceDir/figures/rf_feature_importance.png')

lgb_inst.plotOriginalFeatureSums('featureImportanceDir/figures/lgb_feature_sums.png')
xgb_inst.plotOriginalFeatureSums('featureImportanceDir/figures/xgb_feature_sums.png')
rf_inst.plotOriginalFeatureSums('featureImportanceDir/figures/rf_feature_sums.png')