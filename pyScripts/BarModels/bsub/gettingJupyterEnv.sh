#!/bin/bash
#BSUB -q short               # Queue name
#BSUB -R "rusage[mem=1500]"     # Requested memory
#BSUB -n 1
#BSUB -J gettingJupyterEnv    # Job name
#BSUB -o pyScripts/BarModels/logs/Output_jupyterinterface-%J.out    # Output file
#BSUB -e pyScripts/BarModels/logs/Error_jupyterinterface-%J.err    # Error file

# Load necessary modules
echo "Loading modules"
module load miniconda
module load JupyterLab/3.1.6-GCCcore-11.2.0
conda activate ml

# Set the directory to your working directory
echo "Setting directory"
cd /home/labs/cssagi/barc/FGS_ML/ML_Project  # Change to your actual directory


#call jupyter lab
jupyter-lab --no-browser --ip="0.0.0.0" --port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Done"
