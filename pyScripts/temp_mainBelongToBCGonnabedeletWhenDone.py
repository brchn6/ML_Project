#%%
#-----------------Import the necessary packages-----------------
# from main import *
import pandas as pd
import numpy as np
import os
from featureImportanceDir.Feature_ImportanceClass import Build_Feature_Importance
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#-----------------load the data so i dont need to run the whole thing-----------------
X_train = pd.read_csv('../data/copula_train_set.csv')
y_train = X_train['readmitted']
X_train = X_train.drop('readmitted', axis=1)
X_train_np = np.load('featureImportanceDir/X_train_np.npy', allow_pickle=True).item()
feature_names = open('featureImportanceDir/feature_names.txt', 'r').read().splitlines()
feature_names = np.array(feature_names)
#-----------------------------Done loading the data----------------------------------

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


# # Plot the feature importance
# lgb_inst.plotFeatureImportance()
# xgb_inst.plotFeatureImportance()
# rf_inst.plotFeatureImportance()

# Calculate the sum of the original features
lgb_sums = lgb_inst.calculateSumOfOriginalFeatures()
xgb_sums = xgb_inst.calculateSumOfOriginalFeatures()
rf_sums = rf_inst.calculateSumOfOriginalFeatures()


# Plot the sum of original features
lgb_inst.plotOriginalFeatureSums()
xgb_inst.plotOriginalFeatureSums()
rf_inst.plotOriginalFeatureSums()


#%%
import re
def ManipulateFeatureImportanceData(feature_importance, feature_names):
    # Create a DataFrame
    df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    
    # Define patterns to remove
    patterns_to_remove = ['>[0-9]+', '[0-9]+-[0-9]+', r'(pipeline-[123]_|_)']

    # Clean up feature names
    for pattern in patterns_to_remove:
        df['feature'] = df['feature'].apply(lambda x: re.sub(pattern, '', x))

    # Group by cleaned feature names and sum importance values
    df = df.groupby('feature')['importance'].sum().reset_index()
    
    return df

# Display all rows
pd.set_option('display.max_rows', 40)

# Assuming lgb_importance and lgb_features are already defined
print(ManipulateFeatureImportanceData(lgb_importance, lgb_features))
print(ManipulateFeatureImportanceData(xgb_importance, xgb_features))
print(ManipulateFeatureImportanceData(rf_importance, rf_features))


#%%
# plot the feature importance df
def plotFeatureImportance(df, title):
    import matplotlib.pyplot as plt
    df = df.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(df['feature'], df['importance'], color='skyblue')
    plt.xlabel('Feature Importance Sum')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()
plotFeatureImportance(ManipulateFeatureImportanceData(lgb_importance, lgb_features), 'LGBM Feature Importance')
plotFeatureImportance(ManipulateFeatureImportanceData(xgb_importance, xgb_features), 'XGBoost Feature Importance')
plotFeatureImportance(ManipulateFeatureImportanceData(rf_importance, rf_features), 'Random Forest Feature Importance')