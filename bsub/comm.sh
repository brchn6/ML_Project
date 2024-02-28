bsub -q gpu-interactive -gpu num=1:gmem=5GB -R rusage[mem=8192] -Is "/bin/bash -l" 

ml load miniconda
conda activate ml

/apps/RH7U2/gnu/miniconda/4.10.3_environmentally/envs/jupyter;jupyter notebook --ip 0.0.0.0 --no-browser