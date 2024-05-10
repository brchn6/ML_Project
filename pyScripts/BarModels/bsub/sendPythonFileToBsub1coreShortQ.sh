#!/bin/bash
FILENAME="/home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/RF_Main_Run_BestParams.py"
FILENAME=$(basename $FILENAME)

#BSUB -q short                          # Queue name
#BSUB -R "rusage[mem=1500]"             # Requested memory
#BSUB -n 1                              # Number of CPU cores
#BSUB -W 24:00                          # Running time (24 hours)
#BSUB -N                                # Send mail when job ends
#BSUB -J RF_Main_Run_BestParams
#BSUB -o pyScripts/BarModels/logs/RF_Main_Run_BestParams%J.out
#BSUB -e pyScripts/BarModels/logs/RF_Main_Run_BestParams%J.err

echo "Loading modules"
module load miniconda || { echo "Failed to load miniconda module"; exit 1; }

echo "Setting up environment"
conda activate ml || { echo "Failed to activate conda environment"; exit 1; }

echo "Creating log directories"
mkdir -p pyScripts/BarModels/logs/

echo "Setting directory to your working directory"
cd /home/labs/mayalab/barc/MSc_studies/ML_Project || { echo "Failed to change directory"; exit 1; }

echo "Running Python script"
python /home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/$FILENAME || { echo "Python script failed"; exit 1; }

echo "Done"
