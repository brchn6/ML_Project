#!/bin/bash

bsub -q gpu-interactive -R rusage[mem=8192] -Is "/bin/bash -l"

module load miniconda

conda activate ml

jupyter-lab --no-browser --ip="0.0.0.0" --port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

