#!/bin/bash

#BSUB -J gans_test          # Job name
#BSUB -o test_out.%J      # Output file name
#BSUB -e test_err.%J       # Error file name
#BSUB -n 1                     # Number of CPU cores to reserve
#BSUB -W 1:00                  # Wall time limit in HH:MM
#BSUB -R "rusage[mem=2GB]"    

# Load Python environment
source activate mlproject

# Run Python script
python preprocessing_pipe_copy.py
