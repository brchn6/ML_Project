conda # ML Course Project

## Overview
The objective of this project is to predict readmission of diabetic patients. This objective was predefined as a binary classification task.
The dataset we worked on is the Diabetes 130-US Hospitals for Years 1999 – 2008 from the UC Irvine ML Repository:
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008.
Description of the original features can be found in the supplemental information of the original research paper (see also data/IDS_mapping.csv).
Description of features that were engineered or grouped for this project can be found below in the files description section.  
  
A detailed overview of the project and a summary of the results are written in Project_Summary.docsx

## Installation
To set up the project environment:
1. Clone the repository:

2. Install required packages:
pip install -r requirements.txt

## Usage
python pyScripts/MainScript.py

## Files description:
Folder / files
#### ML_Project/
Data_preparation/Feature_engineering.xlsx : description of the grouping / feature engineering that was done as part of the preprocessing.  
EDA/EDA.ipynb  
PDFs/ PDFs that are the project description. Given by the course host.  
PO/ **** ADD YOUR DESCRIPTION HERE*****  
bsub/ **** ADD YOUR DESCRIPTION HERE*****  
data/ *.csv files that were used in this project. diabetic_data.csv is the raw data.  
pyScripts/  

## Contributing

## License

## Contact


## Check Work Flow 
root *./ML_Project*
1. The first part in this project is to upload the data into the machine environment 
2. data store at ./data/diabetes+130-us+hospitals+for+years+1999-2008.zip
3. The first py script to examine located at EDA/EDA.ipynb ,    
        This file is a well-documented guide that addresses all the crucial questions that need to be answered before starting to deal with ML training or data handling.
4. The 




```markdown
# ML Course Project: Diabetes Readmission Prediction

## Overview
The goal of this project is to predict the readmission of diabetic patients using data from the Diabetes 130-US Hospitals dataset spanning the years 1999 – 2008. This task is framed as a binary classification problem. The dataset was obtained from the UC Irvine Machine Learning Repository, which can be found [here](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).

The project includes comprehensive data preprocessing, feature engineering, and the application of advanced machine learning models for predictive analytics. Detailed descriptions of the original and engineered features are provided in the Data section below.

For a full project report and summary of results, refer to `Project_Summary.docx`.

## Installation
To set up and run the project environment:
1. Clone the repository.
2. Install required Python packages:
   ```bash
   conda install -f environment.yml
   ```

## Usage
Execute the main script to start the project:
```bash
python pyScripts/MainScript.py
```

## File Structure
```
ML_Project/
├── Data_preparation/
│   └── Feature_engineering.xlsx  # Describes the feature engineering process
├── EDA/
│   └── EDA.ipynb                 # Exploratory Data Analysis notebook
├── PDFs/
│   └── Project_description.pdf   # Description documents provided by course hosts
├── data/
│   ├── diabetes_data.csv         # Original dataset
│   ├── IDS_mapping.csv           # Description of original features
│   ├── copula_train_set_300_epochs_4_numeric.csv  # Balanced dataset
│   └── score_table.csv           # Classifier performance estimates
├── pyScripts/
│   ├── ConditionalTransformer.py # Applies transformers conditionally
│   ├── CopulaGenerator.py        # Synthetic data generator
│   ├── evaluation_classifiers.py # Evaluates classifier performance
│   └── generate_n_test_df.py     # Generates and evaluates datasets
├── GuyTrain/
│   ├── X_test_df.csv             # Test dataset
│   ├── X_train_df.csv            # Training dataset
│   ├── xgboost_gridcv.py         # XGBoost tuning with grid search CV
│   └── xgboost_optuna.py         # XGBoost tuning with Optuna
└── deadendscript/
    ├── disease_ids_conds.py      # Functions for data recategorization
    └── feature_importance_rnd_clf.py # Feature importance for random forest
```

```
ML_Project/
├── Data_preparation
│   └── Feature_engineering.xlsx   # Describes the feature engineering process
├── EDA
│   └── EDA.ipynb                  # Exploratory Data Analysis notebook
├── PDFs
│   ├── AMLLS Project Description.pdf # Description documents provided by course hosts
│   └── AMLLS Project.pdf             # Description documents provided by course hosts
├── PO
│   ├── final_step_workflow.pptx      # Descriptive presentation for the colaborators 
│   └── workflow.pptx                 # Full workflow
├── ##Project_Summary.docx            #
├── README.md
├── README_guy.md
├── bsub
│   ├── SMOTE.py.sh
│   ├── comm.sh
│   ├── run_preprocessing.sh
│   └── test_models.sh
├── data
│   ├── IDS_mapping.csv
│   ├── LGBM_top10_features.png
│   ├── copula_train_set_300_epochs_4_numeric.csv
│   ├── diabetes+130-us+hospitals+for+years+1999-2008.zip
│   ├── diabetic_data.csv
│   ├── results_importance_lgbm.csv
│   ├── results_importance_transposed_lgbm.csv
│   ├── results_lgbm.csv
│   └── score_table.csv
├── environment.yml
└── pyScripts
    ├── AddRootDirectoriesToSysPath.py
    ├── BarModels
    │   ├── RF_Main_Run_BestParams.py
    │   ├── RF_Main_Run_FullScript.py
    │   ├── Rendom_forest_BC.py
    │   ├── X_test_df.csv
    │   ├── X_train_df.csv
    │   ├── __main__.py
    │   ├── bsub
    │   │   ├── JupyterHub_Servers.sh
    │   │   ├── RF_Main_Run_FullScript.sh
    │   │   ├── dd.sh
    │   │   ├── getjupitryenv.sh
    │   │   ├── gettingJupyterEnv.sh
    │   │   ├── main.sh
    │   │   ├── sendPythonFileToBsub1coreShortQ.sh
    │   │   ├── simplebsub.sh
    │   │   └── test.sh
    │   ├── logs
    │   │   ├── Error_RF_Main_Run-667086.err
    │   │   ├── Error_RF_Main_Run-667311.err
    │   │   ├── Output_MainFile-815663.out
    │   │   ├── Output_RF_Main_Run-664323.out
    │   │   ├── Output_RF_Main_Run-667086.out
    │   │   └── Output_RF_Main_Run-667311.out
    │   ├── personalClass
    │   │   ├── GetXYstes.py
    │   │   ├── GridSAndXgboostClass.py
    │   │   ├── Rendom_forest_classification_BC_useing_Optuna.py
    │   │   ├── __init__.py
    │   │   ├── feature_importance_script.py
    │   │   └── getinfo.py
    │   ├── results
    │   │   ├── Thumbs.db
    │   │   ├── feature_importance_plot.png
    │   │   ├── feature_importance_table.csv
    │   │   └── prediction_table.csv
    │   ├── y_test.csv
    │   └── y_train.csv
    ├── DefPipeLineClasses.py
    ├── GuyTrain
    │   ├── X_test_df.csv
    │   ├── X_test_np.npy
    │   ├── X_train_df.csv
    │   ├── X_train_np.npy
    │   ├── __init__.py
    │   ├── diabetic_data.csv
    │   ├── feature_importance_script.py
    │   ├── xgboost_gridcv.py
    │   ├── xgboost_optuna.py
    │   ├── xgboost_train_grid_gpu.ipynb
    │   ├── y_test.csv
    │   ├── y_test.npy
    │   ├── y_train.csv
    │   └── y_train.npy
    ├── LGBM.py
    ├── RunPipe.py
    ├── __init__.py
    ├── catboost_info
    │   ├── catboost_training.json
    │   ├── learn
    │   │   └── events.out.tfevents
    │   ├── learn_error.tsv
    │   └── time_left.tsv
    ├── classes
    │   ├── ConditionalTransformer.py
    │   ├── CopulaGenerator.py
    │   ├── SeeTheData.py
    │   ├── __init__.py
    │   └── evaluation_classes.py
    ├── deadendscript
    │   ├── __init__.py
    │   ├── disease_ids_conds.py
    │   ├── feature_importance_rnf_clf.py
    │   ├── generate_n_test_df.py
    │   └── graphs
    │       ├── Thumbs.db
    │       └── feature_importance_15_seeds_mean.png
    ├── featureImportanceDir
    │   ├── Feature_ImportanceClass.py
    │   ├── LGBM_feature_importance.png
    │   ├── Random Forest_feature_importance.png
    │   ├── Thumbs.db
    │   ├── XGBoost_feature_importance.png
    │   ├── X_train.csv
    │   ├── X_train_np.npy
    │   ├── __init__.py
    │   ├── feature_names.txt
    │   ├── temp_mainBelongToBCGonnabedeletWhenDone.py
    │   └── y_train.csv
    ├── main.py
    ├── prepare_data.py
    └── preprocessing_pipe.py
```










## Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact
[Your Name] - [Your Email]
Project Link: [GitHub Repository URL]

## Workflow Check
1. Data upload and initial preprocessing occur within the `Data_preparation/` directory.
2. EDA is conducted in `EDA/EDA.ipynb`, providing insights necessary for model building.
3. Feature engineering details are documented in `Data_preparation/Feature_engineering.xlsx`.

Ensure all paths and file references are accurate and reflect your project's structure. Replace placeholders with specific information as required (e.g., GitHub URL, your contact details).

This README structure will help guide your professor through the project, ensuring clarity on each component's purpose and location.
```

This README structure is modular, detailed, and scalable, covering all aspects of your project. Make sure to update any placeholders with the actual data, links, or personal information before finalizing.


























lets built a README for my project:

this is the README of guy (a collaborator):
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




and this is the raw README that I need to write for this projrct:
conda # ML Course Project

## Overview
The objective of this project is to predict readmission of diabetic patients. This objective was predefined as a binary classification task.
The dataset we worked on is the Diabetes 130-US Hospitals for Years 1999 – 2008 from the UC Irvine ML Repository:
https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008.
Description of the original features can be found in the supplemental information of the original research paper (see also data/IDS_mapping.csv).
Description of features that were engineered or grouped for this project can be found below in the files description section.  
  
A detailed overview of the project and a summary of the results are written in Project_Summary.docsx

## Installation
To set up the project environment:
1. Clone the repository:

2. Install required packages:
pip install -r requirements.txt

## Usage
python pyScripts/MainScript.py

## Files description:
Folder / files
#### ML_Project/
Data_preparation/Feature_engineering.xlsx : description of the grouping / feature engineering that was done as part of the preprocessing.  
EDA/EDA.ipynb  
PDFs/ PDFs that are the project description. Given by the course host.  
PO/ **** ADD YOUR DESCRIPTION HERE*****  
bsub/ **** ADD YOUR DESCRIPTION HERE*****  
data/ *.csv files that were used in this project. diabetic_data.csv is the raw data.  
pyScripts/  

## Contributing

## License

## Contact


## Check Work Flow 
root *./ML_Project*
1. The first part in this project is to upload the data into the machine environment 
2. data store at ./data/diabetes+130-us+hospitals+for+years+1999-2008.zip
3. The first py script to examine located at EDA/EDA.ipynb ,    
        This file is a well-documented guide that addresses all the crucial questions that need to be answered before starting to deal with ML training or data handling.
4. The 




help me build the final one main README file that is documented this project and will assist the teacher to validate that we follow all the steps