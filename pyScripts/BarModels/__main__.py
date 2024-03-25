#%%
"""
Main file for the BarModels directory
"""
%load_ext autoreload
%autoreload 2

#---------------------------- Imports -------------------------------
import numpy as np
import os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# ---------------------------- data incoming -------------------------------
X_train_np = np.load("X_train_np.npy", allow_pickle=True).item()
y_train = np.load("./y_train.npy")
y_test = np.load("./y_test.npy")

# ---------------------------- Rendom_forest -------------------------------
#create a Rendom_forest_classification_BC object
from Rendom_forest import Rendom_forest_classification_BC_defultParams
rf = Rendom_forest_classification_BC_defultParams(X_train_np, y_train, X_train_np, y_train)
#build the model
classifier_fit = rf.build_RandomForestClassifier()
#predict the model
predictions = rf.predict_RandomForestClassifierTrainData(classifier_fit)
#check the accuracy
accuracy = rf.accuracy_score(predictions)
print(accuracy)
print(predictions)
#%%
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train, predictions))
print(recall_score(y_train, predictions))

from sklearn.metrics import f1_score

f1_score(y_train, predictions)

y_scores = classifier_fit.predict_proba(X_train_np)[:, 1]
y_scores


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)
print(fpr, tpr, thresholds)
# plot
import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown
plot_roc_curve(fpr, tpr)