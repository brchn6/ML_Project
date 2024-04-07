#!/bin/bash
#BSUB -q short               # Queue name
#BSUB -R "rusage[mem=16000]"     # Requested memory
#BSUB -J main           # Job name
#BSUB -o pyScripts/BarModels/logs/Output_MainFile-%J.out    # Output file
#BSUB -e pyScripts/BarModels/logs/Error_MainFile-%J.err    # Error file


# Load necessary modules
echo "Loading modules"
module load miniconda

# Set up your environment
echo "Setting up environment"
conda activate ml

# set directory to your working directory
# cd /home/labs/cssagi/barc/FGS_ML/ML_Project

# Run your Python script
echo "Running Python script"
python pyScripts/BarModels/__main__.py
echo "Done"
