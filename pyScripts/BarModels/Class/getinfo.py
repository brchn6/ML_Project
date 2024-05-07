#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
try:
    dir= "/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/results"
    feature_importance_table= pd.read_csv(os.path.join(dir, 'feature_importance_table.csv'))
    feature_importance_table= feature_importance_table.drop('Unnamed: 0', axis=1)
    # feature_importance_table = feature_importance_table.iloc[:, 0:15]
    fig, ax = plt.subplots(figsize=(12, 8))
    # Create a bar plot
    mean_importances = feature_importance_table.mean()
    sns.barplot(x=mean_importances.index, 
                y=mean_importances.values, 
                ax=ax, 
                order=mean_importances.sort_values(ascending=False).index)
    # Additional plot formatting
    ax.set_title('Mean Feature Importances')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
    plt.tight_layout()
    plt.savefig(os.path.join(here, 'results', 'feature_importance_plot.png'))
except Exception as e:
    print(f"Error plotting feature importance: {e}")
    raise



#%%
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
    plt.tight_layout()

X_train= pd.read_csv(os.path.join(here, 'X_train_df.csv'))
X_train = X_train.drop(columns=['diag_3_365.44', 'repaglinide_Down'], axis=1)
featureImportancePlot(X_train.columns, feature_importance_table)
plt.savefig(os.path.join(here, 'results', 'feature_importance_plot.png'))

#%%
#print all the parameters of the model
print(rf_best.get_params())



#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
here = os.path.dirname(os.path.abspath(__file__))
dir= "/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/results"
prediction_table= pd.read_csv(os.path.join(dir, 'prediction_table.csv'))


data= prediction_table.iloc[:, 1:-1]
stv= data.std()
mn= data.mean()

dfata= pd.DataFrame(columns=["score", "mean", "std", 'topat'])
dfata['score']= data.columns
dfata['mean']= mn.values
dfata['std']= stv.values
dfata['topat']= dfata.apply(lambda x: f"{x['mean']:.3f}±{x['std']:.6f}", axis=1)
#transpose the data
dfata = dfata.T
dfata