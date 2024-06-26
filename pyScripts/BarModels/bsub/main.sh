#BSUB -q short               # Queue name
#BSUB -R "rusage[mem=1000]"     # Requested memory
#BSUB -n 1                  # Number of CPU cores
#BSUB -W 24:00              # Running time
#BSUB -N                    # Send mail when job ends
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
python /home/labs/mayalab/barc/MSc_studies/ML_Project/pyScripts/BarModels/__main__.py
echo "Done"
