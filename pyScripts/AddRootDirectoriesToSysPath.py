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

def main():
    # Define the root directory
    root_directory = "/home/labs/cssagi/barc/FGS_ML/ML_Project"
    
    # Add all directories within the root directory to sys.path
    add_directories_to_sys(root_directory)

if __name__ == "__main__":
    main()