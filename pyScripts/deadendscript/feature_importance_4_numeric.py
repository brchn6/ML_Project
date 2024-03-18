#%%
"""
This script was used to test the feature importance on the original train set.
The version of the train set used was the one with 4 numeric features 'num_lab_procedures' and 'num_medications'.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RunPipe import *
import matplotlib.pyplot as plt
from deadendscript.disease_ids_conds import transform_feature_names

#CAll train_set from RunPipe.py:
train_set = train_set

#Redfine the functions from deadendscript/disease_ids_conds.py:
#We want number_emeergency and number_outpatient to be treated as numeric features so we excluded them from the functions:
functions = {
    'time_in_hospital': timeInHosp,
    'num_procedures': numProcedures,
    'number_inpatient': inPatiant,
    'number_diagnoses': numDiag,
    'race' : raceChange,
    'age': ageFunc
}

#Define the preprocessing pipeline:
preprocessing = make_pipeline(
    DropColumns(columns_to_drop),
    IDSTransformer(),
    CustomTransformer(bool_functions),
    ConditionalTransformer(condition=False, transformer=None),
    DiseaseConverter(),
    A1CTransformer(),
    CustomTransformer(functions),
)

#fit and transform the train_set:
train_set = preprocessing.fit_transform(train_set)

#change dtype of number_emergency and number_outpatient to float64:
cols_to_change = ['number_emergency', 'number_outpatient']

#We need to change them to float in order to diffrientiate them from the other numeric features:
train_set[cols_to_change] = train_set[cols_to_change].astype('float64')

#Define the transformer for the NEW numeric features:
#The transformer will not perform standard scaling on the new numeric features:
num_transformer_none = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))])

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

#run col_processor (pipeline from RunPipe.py) on X_train to get X_train_np
X_train_np = col_processor.fit_transform(X_train)

#extract the feature names from the col_processor:
feature_names = col_processor.get_feature_names_out()

#Use function to transform the feature names:
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
plt.show()

#export the feature the plot:
plt.savefig('feature_importance_4_numeric.png')


