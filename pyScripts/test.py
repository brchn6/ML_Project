
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RunPipe import col_processor
import matplotlib.pyplot as plt

#call the X_train from csv name copula_train_set.csv
X_train = pd.read_csv('../data/copula_train_set.csv')
#extract the lables
y_train = X_train['readmitted']
#remove labales from X_train
X_train = X_train.drop('readmitted', axis=1)
#run col_processor (pipeline from RunPipe.py) on X_train to get X_train_np
X_train_np = col_processor.fit_transform(X_train)

feature_names = col_processor.get_feature_names_out()

feature_names = [name[12:] for name in feature_names]

#A function which will return the feature name up to the last '_' in the string:
def get_feature_name(name):
    return name[:name.rfind('_')]

#apply to all the feature names:
feature_names = list(map(get_feature_name, feature_names))

#original feature names:
original_feature_names = X_train.columns


#change age_Older in the feature names to age, change 'admission_type_id_trauma' to 'admission_type_id'
feature_names = [name if name != 'age_Older' else 'age' for name in feature_names]
#change 'admission_type_id_trauma' to 'admission_type_id':
feature_names = [name if name != 'admission_type_id_trauma' else 'admission_type_id' for name in feature_names]
# change  'discharge_disposition_id_discharged', 'discharge_disposition_id_home', 'discharge_disposition_id_left' to 'discharge_disposition_id':
feature_names = [name if name != 'discharge_disposition_id_discharged' else 'discharge_disposition_id' for name in feature_names]
feature_names = [name if name != 'discharge_disposition_id_home' else 'discharge_disposition_id' for name in feature_names]
feature_names = [name if name != 'discharge_disposition_id_left' else 'discharge_disposition_id' for name in feature_names]
#change  'admission_source_id_medical' to 'admission_source_id':
feature_names = [name if name != 'admission_source_id_medical' else 'admission_source_id' for name in feature_names]

feature_names = [name if name != 'num' else 'num_medications' for name in feature_names]

feature_names = [name if name != 'num_lab' else 'num_lab_procedures' for name in feature_names]

feature_names = [name if name != 'chang' else 'change' for name in feature_names]

feature_names = [name if name != 'diabetesMe' else 'diabetesMed' for name in feature_names]


#check that all original feature names are in the feature names:
for name in original_feature_names:
    if name not in feature_names:
        print(name)


model = RandomForestClassifier(random_state=42)

model.fit(X_train_np, y_train)

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

