"""
This script was used to test the feature importance on the copulaGANS generated balanced train set.
The version of the train set used was the one with 4 numeric features 'num_lab_procedures', 'num_medications', 'number_emergency' and number_outpatient'.
"""

#Import the necessary libraries:
import os
#from RunPipe import *
#from deadendscript.synthetic_data_test import *
from classes.evaluation_classes import *
from DefPipeLineClasses import *
from classes.CopulaGenerator import *
from DefPipeLineClasses import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector as selector


#Read the transformed train_set from the csv file:
train_set = pd.read_csv('../data/copula_train_set_300_epochs_4_numeric.csv')

def convert_to_float64(dataframe, columns):
    dataframe[columns] = dataframe[columns].astype('float64')
    return dataframe

cols_to_change = ['number_emergency', 'number_outpatient']

#Define the transformer for the NEW numeric features:
num_transformer_none = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

train_set = convert_to_float64(train_set, cols_to_change)

#Define the transformer for the original numeric features:
col_processor = make_column_transformer(
    (num_transformer_none, selector(dtype_include="float64")),
    (num_transformer, selector(dtype_include="int64")),
    (bool_transformer, selector(dtype_include="bool")),
    (cat_transformer, selector(dtype_include="object")),
    n_jobs=3
)

#fit and transform the train_set:
X_train = train_set.drop('readmitted', axis=1)
y_train = train_set['readmitted']

X_train_np = col_processor.fit_transform(X_train)

#extract the feature names from the col_processor:
feature_names = col_processor.get_feature_names_out()

#original feature names:
original_feature_names = X_train.columns

#Transform the feature names:
feature_names = transform_feature_names(feature_names)

#Manualy change the first feature name to number_outpatient and the second to number_emergency:
feature_names[0] = 'number_outpatient'
feature_names[1] = 'number_emergency'


num_seeds = 15

sums_df = pd.DataFrame()
for seed in range(num_seeds):

    #Define the model:
    model = RandomForestClassifier(random_state=seed)

    #fit the model to the train set:
    model.fit(X_train_np, y_train)

    #extract the feature importances:
    importances = model.feature_importances_

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]

    #sum the importances according names in feature_names:
    sums = {}
    for i in range(len(feature_names)):
        name = feature_names[i]
        if name in sums:
            sums[name] += importances[i]
        else:
            sums[name] = importances[i]
    
    #sort the sums in descending order:
    #sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))

    for k, v in sums.items():
        sums_df.loc['seed ' + str(seed),k] = round(v, 7)
    
    #sort the sums in descending order:
    #sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))

#export sums_df to csv:
sums_df.to_csv('feature_importance_gans_4_numeric.csv')

#%%
#read feature importance from csv:
sums_df = pd.read_csv('fe_seeds_table.csv', index_col=0)


#mean the sums_df:
sums_df_mean = sums_df.mean(axis=0)


#sort the sums_df_mean in decenting order:
sums_df_mean = sums_df_mean.sort_values(ascending=False)

#plot it:
plt.figure(figsize=(10, 10))
plt.barh(sums_df_mean.index, sums_df_mean.values)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Mean Feature Importance of 15 seeds')

#export the feature the plot:z
plt.savefig('feature_importance_15_seeds_mean.png')

plt.show()


"""
#get bottom 10 features:
bottom_10 = dict(list(sums.items())[-10:])

for k, v in bottom_10.items():
    bottom_10[k] = round(v, 7)

#Define the features to drop:
cols_to_drop = [k for k, v in bottom_10.items() if v < 0.0005]

"""




# %%
