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
│   ├── final_step_workflow.pptx      # Descriptive presentation for the collaborators 
│   └── workflow.pptx                 # Full workflow
├── Project_Summary.docx              # Full and informative written descriptive oif this project
├── README.md                         # README file
├── bsub                              # Bsub - bash scripting directory for the use of wexac user only
|
├── data
│   ├── IDS_mapping.csv                                     #This file contains mapping information for IDS (Intrusion Detection System) data.
│   ├── LGBM_top10_features.png                             #This file is an image showing the top 10 features selected by the LightGBM model.
│   ├── copula_train_set_300_epochs_4_numeric.csv           #This file contains a training dataset generated using the copula method with 300 epochs and 4 numeric variables.
│   ├── diabetes+130-us+hospitals+for+years+1999-2008.zip   #original data zip
│   ├── diabetic_data.csv                                   #This file contains the main dataset for the ML project, which includes information about diabetic patients.
│   ├── results_importance_lgbm.csv                         #This file contains the feature importance scores generated by the LightGBM model.
│   ├── results_importance_transposed_lgbm.csv              #This file contains the transposed feature importance scores generated by the LightGBM model.
│   ├── results_lgbm.csv                                    #This file contains the results of the ML model trained using LightGBM.
│   └── score_table.csv                                     #This file contains a table of scores related to the ML project.
|
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
    │   ├── X_test_df.csv                   # Final train and test sets used for training the xgboost model.   
    │   ├── X_test_np.npy                   # Final train and test sets used for training the xgboost model.   
    │   ├── X_train_df.csv                  # Final train and test sets used for training the xgboost model.
    │   ├── X_train_np.npy                  # Final train and test sets used for training the xgboost model.
    │   ├── __init__.py                     # 
    │   ├── diabetic_data.csv               # Fulldata
    │   ├── feature_importance_script.py    # Generates feature importance plot for the xgboost model post-tuning.                   
    │   ├── xgboost_gridcv.py               # Hyperparameter tuning for xgboost using grid search cross-validation.       
    │   ├── xgboost_optuna.py               # Hyperparameter tuning for xgboost using Optuna.  
    │   ├── xgboost_train_grid_gpu.ipynb    # Hyperparameter tuning for xgboost using grid search cross-validation.                   
    │   ├── y_test.csv                      # Final train and test sets used for training the xgboost model.
    │   ├── y_test.npy                      # Final train and test sets used for training the xgboost model.
    │   ├── y_train.csv                     # Final train and test sets used for training the xgboost model.
    │   └── y_train.npy                     # Final train and test sets used for training the xgboost model.
    ├── LGBM.py                             # Saar main script                           
    ├── RunPipe.py                          # python file for running the pipeline
    ├── __init__.py                         
    |
    ├── classes
    │   ├── ConditionalTransformer.py
    │   ├── CopulaGenerator.py
    │   ├── SeeTheData.py
    │   ├── __init__.py
    │   └── evaluation_classes.py
    ├── deadendscript
    │   ├── __init__.py
    │   ├── disease_ids_conds.py            #This script contains function used for recategorizing the data in the dataset, as well as to store long lists usedto gnerate the dataset.
    │   ├── feature_importance_rnf_clf.py   #This script was used to test the feature importance on the copulaGANS generated balanced train set on 15 different seeds.        
    │   ├── generate_n_test_df.py           #
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


## Contact
Bar Cohen - bar.cohen@weizmann.ac.il
Project Link: [GitHub Repository URL]

## Workflow Check
1. Data upload and initial preprocessing occur within the `Data_preparation/` directory.
    The data folder contains all the csv files given to us at the beginning of the project as well as the final train set used for trqaining the model:
    1. diabetes_data.csv
    2. IDS_mapping.csv
    3. copula_train_set_300_epochs_4_numeric.csv

2. EDA is conducted in `EDA/EDA.ipynb`, providing insights necessary for model building.
3. Feature engineering details are documented in `Data_preparation/Feature_engineering.xlsx`.
4. #pyScripts folder: 
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

    #BarModels:
    ```
    1. __main__.py  is the main python script that run the classes build in Rendom_forest_BC.py file
    2. Rendom_forest_BC.py hold the classes use in the main script
    3. RF_Main_Run_BestParams.py is the script that run the best parameters for the random forest model
    4. RF_Main_Run_FullScript.py is the script that run the full script for the random forest model
    5. bsub is a directory that hold the bash scripts that run the python scripts on the cluster
    6. logs is a directory that hold the logs of the scripts that run on the cluster
    7. personalClass is a directory that hold the classes that are used in the main script
    8. results is a directory that hold the results of the model
    9. X_test_df.csv , X_train_df.csv , y_test.csv , y_train.csv are the final train and test sets used for training the model
    ```

    #featureImportanceDir:
    ```
    1. Feature_ImportanceClass.py - This class is used to generate the feature importance plot of the dataset using the random forest classifier.
    2. LGBM_feature_importance.png - This image shows the feature importance scores generated by the LightGBM model.
    3. Random Forest_feature_importance.png - This image shows the feature importance scores generated by the Random Forest model.
    4. XGBoost_feature_importance.png - This image shows the feature importance scores generated by the XGBoost model.
    ```

    #classes:
    ```
    1. ConditionalTransformer.py - This class is used to conditionally apply a transformer to the data.
    If the condition is set to True, the transformer is applied to the data, else the data is returned as is.
    This class was used in the pipline in order to make the GANS class conditional