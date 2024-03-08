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

# --------------------------------------Rendom_forest regression--------------------------------------
def build_RandomForestRegressor (train_features,train_labels):
    # Create a random forest regressor object 
    # Instantiate model with 1 decision trees
    regressor = RandomForestRegressor(n_estimators=1, random_state=42)
    
    # Train the model on training data
    regressor = regressor.fit(train_features, train_labels)
    return regressor

def predict_RandomForestRegressor (regressor, test_features):
    # Use the forest's predict method on the test data
    predictions = regressor.predict(test_features)
    return predictions


# --------------------------------------Rendom_forest classification--------------------------------------
def build_RandomForestClassifier (train_features,train_labels):
    # Create a random forest classifier object
    # Instantiate model with 10 decision trees
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Train the model on training data
    classifier_fit = classifier.fit(train_features, train_labels)
    return classifier_fit


def predict_RandomForestClassifier (classifier, test_features):
    # Use the forest's predict method on the test data
    predictions = classifier.predict(test_features)
    return predictions



# --------------------------------------Rendom_forest Regression Class--------------------------------------
class Rendom_forest_regression_BC:
    def __init__(self, train_features,train_labels,test_features,test_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
    def build_RandomForestRegressor (self):
        # Create a random forest regressor object 
        # Instantiate model with 1 decision trees
        regressor = RandomForestRegressor(n_estimators=1, random_state=42)
        
        # Train the model on training data
        regressor = regressor.fit(self.train_features, self.train_labels)
        return regressor
    
    def predict_RandomForestRegressor (self, regressor):
        # Use the forest's predict method on the test data
        predictions = regressor.predict(self.test_features)
        return predictions