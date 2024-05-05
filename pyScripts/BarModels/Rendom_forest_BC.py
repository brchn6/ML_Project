def random_forest_script():
    """
    This script demonstrates the usage of a random forest algorithm for classification or regression tasks.

    Random forests are an ensemble learning method that combines multiple decision trees to make predictions.
    They are commonly used for tasks such as classification, regression, and feature selection.

    This script assumes that you have already preprocessed your data and split it into training and testing sets.

    
            !!!!!THIS SCRIPT IS NOT CALLABLE, FOR RUN IT YOU NEED TO CALL FROM THE MAIN SCRIPT!!!
    Steps:
    1. Import the necessary libraries.
    2. Load the training and testing data.
    3. Create a random forest classifier or regressor object.
    4. Fit the model to the training data.
    5. Make predictions on the testing data.
    6. Evaluate the model's performance using appropriate metrics.
    7. Optionally, tune the hyperparameters of the random forest model.
    8. Repeat steps 4-7 as needed.

    Note: This script serves as a template and may need to be modified based on your specific use case.

    Example usage:
    random_forest_script()

    
    Aouther: Barc
    Date: 2024/08/03
    """
#---------------------------------------Importing the necessary libraries---------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold

from sklearn.metrics import log_loss , f1_score ,accuracy_score, roc_auc_score

from sklearn.metrics import confusion_matrix

import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# --------------------------------------Rendom_forest Regression Class--------------------------------------
"""class Rendom_forest_regression_BC:
    def __init__(self, train_features, train_labels, test_features, test_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
    
    def build_RandomForestRegressor(self):
        # Create a random forest regressor object 
        # Instantiate model with 1 decision tree
        regressor = RandomForestRegressor(n_estimators=1, random_state=42)
        
        # Train the model on training data
        regressor = regressor.fit(self.train_features, self.train_labels)
        return regressor
    
    def predict_RandomForestRegressor(self, regressor):
        # Use the forest's predict method on the test data
        predictions = regressor.predict(self.test_features)
        return predictions
"""    
# --------------------------------------Rendom_forest Classification Class--------------------------------------
class Rendom_forest_classification_BC_defultParams:
    def __init__(self, np_train_features, train_labels, np_test_features, test_labels):
        self.train_features = np_train_features
        self.train_labels = train_labels
        self.test_features = np_test_features
        self.test_labels = test_labels
    
    def build_RandomForestClassifier (self):
        """
        # Create a random forest classifier object
        # Instantiate model with 10 decision trees
        Returns:
        classifier_fit: the trained model
        """
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train the model on training data
        classifier_fit = classifier.fit(self.train_features, self.train_labels)
        return classifier, classifier_fit
    
    #set a prediction metho for the train data
    def predict_RandomForestClassifierTrainData (self, classifier):
        # Use the forest's predict method on the test data
        predictions = classifier.predict(self.train_features)
        return predictions
    
    def predict_RandomForestClassifierTestData(self, classifier):
        # Use the forest's predict method on the test data
        predictions = classifier.predict(self.test_features)
        return predictions

    def predict_RandomForestClassifierTrainData_proba (self, classifier):
        # Use the forest's predict method on the test data
        predictions_proba = classifier.predict_proba(self.train_features)
        return predictions_proba[:,1]

    def predict_RandomForestClassifierTestData_proba(self, classifier):
        # Use the forest's predict method on the test data
        predictions_proba = classifier.predict_proba(self.test_features)
        return predictions_proba[:,1]
    
    #build a accuracy score method
    def accuracy_score(self, predictions,data):
        """
        Returns:
        accuracy: the accuracy of the model
        """
        accuracy = accuracy_score(data, predictions)
        f1_weighted= f1_score(data, predictions, average='weighted')
        f1_binary= f1_score(data, predictions, average='binary')
        return accuracy, f1_weighted, f1_binary
    
    #build a accuracy score method
    def accuracy_score_proba(self, predictions_proba ,data):
        """
        Returns:
        accuracy: the accuracy of the model
        """
        logLoss = log_loss(data, predictions_proba)
        roc_auc = roc_auc_score(data, predictions_proba)
        return logLoss, roc_auc
    
    def get_params(self, classifier):
        return classifier.get_params()
    
    def make_confusion_matrix(self, predictions, data):
        return confusion_matrix(data, predictions)

class Rendom_forest_classification_BC_useingGridSearchCV:
    def __init__(self, np_train_features, train_labels, np_test_features, test_labels):
        self.train_features = np_train_features
        self.train_labels = train_labels
        self.test_features = np_test_features
        self.test_labels = test_labels
        self.best_params = None  # Initialize best_params attribute


    def gridSearchCV_RandomForestClassifier(self):
        # Define the parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        return param_grid
        
    def build_RandomForestClassifierWithGridSearchCV(self):
        # Create a Random Forest classifier object
        classifier = RandomForestClassifier(random_state=42)
        
        # Create GridSearchCV object with the classifier and parameter grid
        grid_search = GridSearchCV(estimator=classifier, param_grid=self.gridSearchCV_RandomForestClassifier(), cv=5, n_jobs=-1, verbose=2)

        #fit the model to the train data
        classifier_fit = grid_search.fit(self.train_features, self.train_labels) 

        # Get the best model
        best_rf_classifier = classifier_fit.best_estimator_
        
        
        #get the best parameters and the roc_auc
        self.best_params = classifier_fit.best_params_
        roc_auc = classifier_fit.best_score_    
        print(f"Best parameters: {self.best_params}")
        print(f"The best parameters in this run are: {self.best_params} and the ROC AUC score is: {roc_auc}")
       
        return classifier, classifier_fit, best_rf_classifier, self.best_params
    
    def update_parameter_grid(self):
        if not self.best_params:
            print("Error: Best params are not set yet.")
            return None

        # Check each parameter individually before use to handle cases where they might be None
        n_estimators = self.best_params.get('n_estimators', 10)  # Default to 10 if not set
        max_depth = self.best_params.get('max_depth', 30)  # Default to 30 if not set
        min_samples_split = self.best_params.get('min_samples_split', 2)  # Default to 2 if not set
        min_samples_leaf = self.best_params.get('min_samples_leaf', 1)  # Default to 1 if not set
        max_features = self.best_params.get('max_features', 'auto')  # Default to 'auto' if not set
        bootstrap = self.best_params.get('bootstrap', True)  # Default to True if not set

        updated_param_grid = {
            'n_estimators': [n_estimators, n_estimators + 50, n_estimators + 100],
            'max_depth': [max_depth, max_depth + 10, max_depth + 20] if max_depth is not None else [None],
            'min_samples_split': [min_samples_split, min_samples_split + 3, min_samples_split + 6],
            'min_samples_leaf': [min_samples_leaf, min_samples_leaf + 1, min_samples_leaf + 2],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [bootstrap, not bootstrap]
        }
        return updated_param_grid


    #set a prediction metho for the train data and the test data
    def predict_RandomForestClassifier(self, classifier):
        # Use the forest's predict method on the test data
        predictions_Train_set= classifier.predict(self.train_features)
        predictions_Train_set_proba= classifier.predict_proba(self.train_features)
        predictions_Test_set= classifier.predict(self.test_features)
        predictions_Test_set_proba= classifier.predict_proba(self.test_features)
        return predictions_Train_set, predictions_Train_set_proba[:,1] , predictions_Test_set, predictions_Test_set_proba[:,1]

    
    #build a accuracy score method
    def accuracy_score(self, predictions, predictions_proba, data):
        """
        Calculate accuracy, weighted F1, binary F1 scores, log loss, and ROC AUC for predictions.

        Parameters:
        - predictions: Predicted labels.
        - predictions_proba: Predicted probabilities.
        - data: True labels.

        Returns:
        - Tuple containing accuracy, f1_weighted, f1_binary, log_loss, and roc_auc scores.
        """
        accuracy = accuracy_score(data, predictions)
        f1_weighted = f1_score(data, predictions, average='weighted')
        f1_binary = f1_score(data, predictions, average='binary')
        log_loss_val = log_loss(data, predictions_proba)
        roc_auc_val = roc_auc_score(data, predictions_proba)

        return accuracy, f1_weighted, f1_binary, log_loss_val, roc_auc_val

    def get_best_params(self):
        return self.best_params
    def make_confusion_matrix(self, predictions, data):
        return confusion_matrix(data, predictions)

