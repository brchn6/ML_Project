#%%
"""
RandomForest.py

This script implements a machine learning model using the Random Forest algorithm. Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It is commonly used for classification and regression tasks.

The script will train a Random Forest model on a given dataset and use it to make predictions on new data. It will handle data preprocessing, feature selection, model training, and evaluation.

Please make sure to have the necessary dependencies installed before running the script. You may need to install scikit-learn or any other required libraries.

Author: Barc
Date: 2024-03-05
"""
# ------------------- Importing Libraries -------------------
from sklearn.ensemble import RandomForestRegressor
# ------------------- Load Data -------------------
"""
here we gonna get the X_train, X_test, y_train, y_test
"""
from RunPipe import *

#%%
# see the X_train_np as a df
# show_data()
#get the labels
train_labels = y_train
train_labels
#%%
# ------------------- setting up the model -------------------
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)