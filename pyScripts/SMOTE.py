
# # Random Forest with SMOTE

# In[104]:


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# PCA
pca = PCA()

# SMOTE
smote = SMOTE(random_state=42)

# Random Forest Classifier
modelRF_PCA_SMOTE = RandomForestClassifier(random_state=42)

# Create a pipeline with PCA, SMOTE, and RF
pipeline_PCA_SMOTE = ImbPipeline([
    ('reduce_dim', pca),
    ('smote', smote),
    ('rf', modelRF_PCA_SMOTE)
])

pipeline_PCA_SMOTE.fit(X_bac_train_dict['oral_species_new_X_bac_train'], y_bac_train_dict['oral_species_new_y_bac_train'].ravel())

predictions_PCA_SMOTE = pipeline_PCA_SMOTE.predict(X_bac_test_dict['oral_species_new_X_bac_test'])
proba_predictions_PCA_SMOTE = pipeline_PCA_SMOTE.predict_proba(X_bac_test_dict['oral_species_new_X_bac_test'])[:,1]

print(classification_report(y_bac_test_dict['oral_species_new_y_bac_test'], predictions_PCA_SMOTE))
print(roc_auc_score(y_bac_test_dict['oral_species_new_y_bac_test'], proba_predictions_PCA_SMOTE))


# In[111]:


from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score

# PCA
pca = PCA()

# SMOTE
smote = SMOTE(random_state=42)

# Random Forest Classifier
modelRF = RandomForestClassifier(random_state=42)

# Create a pipeline with PCA, SMOTE, and RF
pipeline = ImbPipeline([
    ('reduce_dim', pca),
    ('smote', smote),
    ('rf', modelRF)
])

# Define the hyperparameter grid
param_grid = {
    'reduce_dim__n_components': [1, 2, 3], 
    'rf__n_estimators': [50, 100], 
    'rf__max_depth': [None, 5, 10],
 
}

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, scoring=make_scorer(roc_auc_score))

# Fit the model to the training data
grid_search.fit(X_bac_train_dict['oral_species_new_X_bac_train'], y_bac_train_dict['oral_species_new_y_bac_train'].ravel())

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

# Evaluate the model on the test data using the best hyperparameters
best_model = grid_search.best_estimator_
predictions_PCA_SMOTE = best_model.predict(X_bac_test_dict['oral_species_new_X_bac_test'])
proba_predictions_PCA_SMOTE = best_model.predict_proba(X_bac_test_dict['oral_species_new_X_bac_test'])[:,1]

print(classification_report(y_bac_test_dict['oral_species_new_y_bac_test'], predictions_PCA_SMOTE))
print(roc_auc_score(y_bac_test_dict['oral_species_new_y_bac_test'], proba_predictions_PCA_SMOTE))

