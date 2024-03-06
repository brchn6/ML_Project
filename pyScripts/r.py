#%%
"""
Rendom forest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from RunPipe import X_train_np , y_train , X_test_np , y_test

rf = RandomForestRegressor(n_estimators = 2, random_state = 42)

rf.fit(X_train_np, y_train)


# Use the forest's predict method on the test data
predictions = rf.predict(X_test_np)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Handle division by zero in MAPE calculation by adding a small constant (e.g., 1e-10)
# This ensures you don't divide by zero, but the adjustment is small enough to not significantly affect the result
mape = 100 * (errors / (y_test + 1e-10))

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


#%%
# plot the learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    