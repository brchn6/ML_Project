#%%
import sys
import os

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

def BetterAddDirectoriesToSys(root_directory):
    # check if the root directory base na me is ML_Project:
    if root_directory.split("/")[-1] == "ML_Project":
        # Get a list of all directories in the root directory
        directories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
        
        # Add each directory to sys.path
        for directory in directories:
            if directory not in sys.path:
                sys.path.append(directory)
    else:
        print("Root directory does not exist.")
    


def AddRootDirectoriesToSys():
    # Define the root directory
    root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add all directories within the root directory to sys.path
    # add_directories_to_sys(root_directory)
    BetterAddDirectoriesToSys(root_directory)
    return root_directory

if __name__ == "__main__":
    AddRootDirectoriesToSys()