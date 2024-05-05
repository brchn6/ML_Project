#!/bin/bash
#BSUB -q gpu-interactive               # Queue name
#BSUB -gpu "num=1:j_exclusive=no
#BSUB -R "rusage[mem=8000]"     # Requested memory
#BSUB -R affinity[thread*1] # affinity
#BSUB -Is /bin/bash