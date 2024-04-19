#!/bin/bash
#BSUB -q gpu-interactive               # Queue name
#BSUB -gpu "num=2:j_exclusive=yes:gmem=8G:gmodel=NVIDIAA40"  # Requested GPU resources
#BSUB -R "rusage[mem=8000]"     # Requested memory
#BSUB -J gettingJupyterEnv    # Job name
#BSUB -o pyScripts/BarModels/logs/Output_jupyterinterface-%J.out    # Output file
#BSUB -e pyScripts/BarModels/logs/Error_jupyterinterface-%J.err    # Error file


# Load necessary modules
echo "Loading modules"
module load miniconda

module load JupyterLab/3.1.6-GCCcore-11.2.0


# Set up your environment
echo "Setting up environment"
conda activate tf

# set directory to your working directory
# cd /home/labs/cssagi/barc/FGS_ML/ML_Project

# Run your Python script
echo "Running Python script"
jupyter-lab --no-browser --ip="0.0.0.0" --port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Done"

