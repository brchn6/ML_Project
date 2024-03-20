"""
This script was used to test the feature importance on the copulaGANS generated balanced train set.
The version of the train set used was the one with 4 numeric features 'num_lab_procedures', 'num_medications', 'number_emergency' and number_outpatient'.
"""
#%%
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
#%%
#Call the train_set from RunPipe.py:
"""
train_set = train_set

#Redfine the functions from deadendscript/disease_ids_conds.py:
functions = {
    'time_in_hospital': timeInHosp,
    'num_procedures': numProcedures,
    'number_inpatient': inPatiant,
    'number_diagnoses': numDiag,
    'race' : raceChange,
    'age': ageFunc
}

#Define copula_gans object:
copula_gans = CopulaGANSyntheticDataGenerator(label_column='readmitted', minority_class_label=0,
                                               majority_class_label=1, gans = 'copula',
                                               boolean_columns=['change', 'diabetesMed'], 
                                               epochs=300)

#Define the preprocessing pipeline:  
preprocessing = make_pipeline(
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
    ConditionalTransformer(condition=True, transformer=copula_gans),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)

#fit and transform the train_set:
df_transformed = preprocessing.fit_transform(train_set)
df_name =  'copula_train_set_300_epochs_4_numeric'

#Export the transformed train_set to a csv file:
path = os.path.join(os.getcwd(), '..', 'data')
csv_filename = df_name + '.csv'
df_transformed.to_csv(os.path.join(path, csv_filename), index=False)


"""

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

cols_to_drop = ['acarbose',
 'miglitol',
 'chlorpropamide',
 'glipizide-metformin',
 'tolazamide',
 'tolbutamide',
 'metformin-pioglitazone',
 'glimepiride-pioglitazone']

"""
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

#Define the model:
model = RandomForestClassifier(random_state=42)

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
sums = dict(sorted(sums.items(), key=lambda item: item[1], reverse=True))


#plot the feature importances according to the sums dictionary (which is sorted):
plt.figure(figsize=(10, 5))
plt.title('Feature Importances')
plt.bar(range(len(sums)), list(sums.values()), align='center')
plt.xticks(range(len(sums)), list(sums.keys()), rotation=90)


#export the feature the plot:
#plt.savefig('feature_importance_gans_4_numeric.jpeg')

plt.show()

#get bottom 10 features:
bottom_10 = dict(list(sums.items())[-10:])

for k, v in bottom_10.items():
    bottom_10[k] = round(v, 7)

#Define the features to drop:
cols_to_drop = [k for k, v in bottom_10.items() if v < 0.0005]
"""
"""
['acarbose',
 'miglitol',
 'chlorpropamide',
 'glipizide-metformin',
 'tolazamide',
 'tolbutamide',
 'metformin-pioglitazone',
 'glimepiride-pioglitazone']
"""
