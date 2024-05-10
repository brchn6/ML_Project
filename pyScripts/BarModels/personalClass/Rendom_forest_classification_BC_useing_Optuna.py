class Rendom_forest_classification_BC_useing_Optuna:
    
    def __init__(self, np_train_features, train_labels, np_test_features, test_labels):
        self.train_features = np_train_features #X_train_np
        self.train_labels = train_labels #y_train
        self.test_features = np_test_features #X_test_np
        self.test_labels = test_labels #y_test


    def objective(self, trial):
        params = { 
                     "max_depth" : trial.suggest_int("max_depth", 1, 20),
                     "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.1),
                     "n_estimators" : trial.suggest_int('n_estimators', 5, 300),
                     'subsample': trial.suggest_float('subsample', 0.01, 1.0, log = True),
                     'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
                     "booster" : trial.suggest_categorical('booster', ['gbtree', 'dart']),
                     "min_child_weight" : trial.suggest_int("min_child_weight", 1, 50), 
                     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log = True),
                     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log = True),
                     "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 0.5, 1),
                     "max_delta_step" : trial.suggest_int("max_delta_step", 0, 10),                     
                     "grow_policy" : trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log = True)
                     }
        # Create a Random Forest classifier object
        with tf.device('/GPU:0'):
            classifier = RandomForestClassifier(**params, renmode_state=42)

        # Use cross-validation to evaluate the model
        with tf.device('/GPU:0'):
            score = cross_val_score(classifier, self.train_features, self.train_labels, cv=3, scoring='roc_auc').mean()

        return score
        
    def optimize_params(self):
        with tf.device('/GPU:1'):
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=1000)

            best_params = study.best_params
            best_rf_classifier = RandomForestClassifier(**best_params)
            best_score = study.best_value

            print(f"the best parameters are: {best_params} and the best score is: {best_score}")
        return best_rf_classifier, best_params
        
    def train_model(self, best_rf_classifier):
        best_classifier = RandomForestClassifier(**best_rf_classifier)
        with tf.device('/GPU:1'):
            best_classifier_fit = best_classifier.fit(self.train_features, self.train_labels)

        return best_classifier_fit
    
    def evaluate_model(self, classifier):
        predictions = classifier.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, predictions)
        f1_weighted = f1_score(self.test_labels, predictions, average='weighted')

        return accuracy, f1_weighted
    
    def build_RandomForestClassifierWithOptuna_tuning(self, trial):
        # Define the parameter grid
        param_grid = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 40, 50]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10, 20]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4, 8, 16]),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        return param_grid

    def build_RandomForestClassifierWithOptuna(self):
        # Create a Random Forest classifier object
        classifier = RandomForestClassifier(random_state=42)
        
        # Create GridSearchCV object with the classifier and parameter grid
        with tf.device('/GPU:0'):
            study = optuna.create_study(direction='maximize')
            study.optimize(self.build_RandomForestClassifierWithOptuna_tuning, n_trials=100)
            best_params = study.best_params
            best_rf_classifier = RandomForestClassifier(**best_params)
        
        #fitt the model to the train data
        with tf.device('/GPU:0'):
            classifier_fit = best_rf_classifier.fit(self.train_features, self.train_labels) 

        return best_rf_classifier, classifier_fit ,best_params 

    #set a prediction metho for the train data
    def predict_RandomForestClassifierTrainData (self, classifier):
        # Use the forest's predict method on the test data
        predictions = classifier.predict(self.train_features)
        return predictions

    def predict_RandomForestClassifierTestData(self, classifier):
        # Use the forest's predict method on the test data
        predictions = classifier.predict(self.test_features)
        return predictions
    
