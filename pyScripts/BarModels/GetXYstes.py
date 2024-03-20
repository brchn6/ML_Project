"""

"""
import os
import numpy as np
def save_data(X_train_np, y_train):
    if not os.getcwd() == "/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts":
        os.chdir("/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts")

    dir_to_save= "./BarModels"

    #save the X_train_np matrix to a file
    np.save(dir_to_save + "/X_train_np", X_train_np)
    #save the y_train matrix to a file
    np.save(dir_to_save + "/y_train", y_train)
    print("Data saved")
    print("X_train_np shape: ", X_train_np.shape)
    print("y_train shape: ", y_train.shape)
