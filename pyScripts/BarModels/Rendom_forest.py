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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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
        return classifier_fit
    
    #set a prediction metho for the train data
    def predict_RandomForestClassifierTrainData (self, classifier):
        # Use the forest's predict method on the test data
        predictions = classifier.predict(self.train_features)
        return predictions
    
    #build a accuracy score method
    def accuracy_score(self, predictions):
        """
        Returns:
        accuracy: the accuracy of the model
        """
        accuracy = accuracy_score(self.train_labels, predictions)
        return accuracy
    
class Rendom_forest_classification_BC_useingGridSearchCV:
    
    def __init__(self, np_train_features, train_labels, np_test_features, test_labels):
        self.train_features = np_train_features
        self.train_labels = train_labels
        self.test_features = np_test_features
        self.test_labels = test_labels

    def gridSearchCV_RandomForestClassifier(self):

        # Define the parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create a Random Forest classifier object
        rf_classifier = RandomForestClassifier(random_state=42)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
        
        # Train the model on training data
        grid_search.fit(self.train_features, self.train_labels)
        
        # Get the best estimator
        best_rf_classifier = grid_search.best_estimator_
        
        return best_rf_classifier