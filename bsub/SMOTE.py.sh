#!/bin/bash
#BSUB -q new-short                  # Queue name
#BSUB -R "rusage[mem=16000]"      # Requested memory
#BSUB -J SMOTE_GPU_JOB            # Job name
#BSUB -n 4                       # Number of CPU cores
#BSUB -threads 4                 # Number of threads
#BSUB -o SMOTE_GPU_JOB_%J.out    # Output file
#BSUB -e SMOTE_GPU_JOB_%J.err    # Error file


# Load necessary modules
module load python  # Adjust based on your environment
module load miniconda

# Set up your environment
conda activate ml

# set directory to your working directory
cd /home/labs/cssagi/barc/FGS_ML/ML_Project

# Run your Python script
python pyScripts/SMOTE.py
