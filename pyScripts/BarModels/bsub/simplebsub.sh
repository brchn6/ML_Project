#!/bin/bash
#BSUB -q interactive                # Queue name
#BSUB -R rusage[mem=8192]     # Requested memory
#BSUB -Is "/bin/bash -l"     # Interactive shell



# Load necessary modules
echo "Loading modules"
module load miniconda

# Set up your environment
echo "Setting up environment"
conda activate ml

