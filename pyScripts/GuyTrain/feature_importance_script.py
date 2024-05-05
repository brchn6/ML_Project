import matplotlib.pyplot as plt
import pandas as pd

def transform_feature_names(feature_names):
    # Function to return the feature name up to the last '_'
    def get_feature_name(name):
        return name[:name.rfind('_')]

    # Apply the get_feature_name function to all feature names
    feature_names = list(map(get_feature_name, feature_names))

    # Replace specific strings in the feature names
    replacements = {
        'age_Older': 'age',
        'admission_type_id_trauma': 'admission_type_id',
        'discharge_disposition_id_discharged': 'discharge_disposition_id',
        'discharge_disposition_id_home': 'discharge_disposition_id',
        'discharge_disposition_id_left': 'discharge_disposition_id',
        'admission_source_id_medical': 'admission_source_id',
        'num': 'num_medications',
        'num_lab': 'num_lab_procedures',
        'chang': 'change',
        'diabetesMe': 'diabetesMed'
    }
    feature_names = [replacements.get(name, name) for name in feature_names]

    return feature_names

def featureImportancePlot (feature_names, table):
    
    table_mean = table.mean(axis = 0)
    table_std = table.std(axis = 0)

    #Transform the feature names:
    feature_names = transform_feature_names(feature_names)

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
    
    # Add labels and title
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.title('Top 10 Feature Importance on (mean of 15 seeds)')
    
    
    #export the feature the plot:z
    # plt.savefig('feature_importance_15_seeds_mean.png')
    
    plt.show()

