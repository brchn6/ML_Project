#%%
"""
This script adds all directories within the root directory to sys.path.
and will be called by the RunPipe.py to add all directories within the root directory to sys.path.
"""
#------------------------------Imports---------------------------------
import sys
import os
#------------------------------Functions---------------------------------
def add_directories_to_sys(root_directory):
    # Check if the root directory exists
    if os.path.exists(root_directory):
        # Get a list of all directories in the root directory
        directories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
        
        # Add each directory to sys.path
        for directory in directories:
            if directory not in sys.path:
                sys.path.append(directory)
    else:
        print("Root directory does not exist.")

def define_data_paths():
    """
    Define the paths for the data files.
    Returns:
    Tuple[str, str]: A tuple containing the paths to the data file and the mapping file.
    """
    # Define the path you want to add
    GETCWD = os.getcwd()

    #------------------------------get the data file ----------------

    if os.path.basename(GETCWD) == "pyScripts":
        PathToData = os.path.join(GETCWD + "/../data/diabetic_data.csv" )
        PathToMap = os.path.join(GETCWD + "/../data/IDS_mapping.csv" )
    #adding a logic to the path of the data file do i could work from any dir
    elif os.path.basename(GETCWD) == "barc":
        PathToData = os.path.join(GETCWD + "/FGS_ML/ML_Project/data/diabetic_data.csv" )
        PathToMap = os.path.join(GETCWD + "/FGS_ML/ML_Project/data/IDS_mapping.csv" )
    elif os.path.basename(GETCWD) == "ML_Project":
        PathToData = os.path.join(GETCWD + "/data/diabetic_data.csv" )
        PathToMap = os.path.join(GETCWD + "/data/IDS_mapping.csv" )
    elif os.path.basename(GETCWD) == "FGS_ML":
        PathToData = os.path.join(GETCWD + "/data/diabetic_data.csv" )
        PathToMap = os.path.join(GETCWD + "/data/IDS_mapping.csv" )
    elif os.path.basename(GETCWD) == "prepdir":
        PathToData = os.path.join(GETCWD + "/../data/diabetic_data.csv" )
        PathToMap = os.path.join(GETCWD + "/../data/IDS_mapping.csv" )
    
    return PathToData, PathToMap

# ---------------------------------End functions--------------------------------



#------------------------------Main---------------------------------

def EnvPrepa_main():
    # Define the root directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    add_directories_to_sys(ROOT_DIR)
    ROOT_DIR = os.path.join(ROOT_DIR, "..")
    add_directories_to_sys(ROOT_DIR)

