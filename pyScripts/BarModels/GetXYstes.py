"""

"""
import os
import numpy as np
def save_data(X_train_np, y_train, X_test_np, y_test, dir_to_save= "./BarModels"):
    if not os.getcwd() == "/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts":
        os.chdir("/home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts")

    #save the X_train_np matrix to a file
    np.save(dir_to_save + "/X_train_np", X_train_np)
    #save the y_train matrix to a file
    np.save(dir_to_save + "/y_train", y_train)
    np.save(dir_to_save + "/X_test_np", X_test_np)
    np.save(dir_to_save + "/y_test", y_test)
    print("Data saved")