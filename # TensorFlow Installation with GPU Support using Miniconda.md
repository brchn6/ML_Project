Sure, here's a Markdown file with the actions described:

```markdown
# TensorFlow Installation with GPU Support using Miniconda

## Installation Steps:

1. Load the Miniconda module:
   ```bash
   ml miniconda
   ```

2. Create a Conda environment named "tf" with Python 3.9:
   ```bash
   conda create --name=tf python=3.9
   ```

3. Activate the "tf" environment:
   ```bash
   conda activate tf
   ```

4. Install CUDA toolkit 11.2.2 and cuDNN 8.1.0 using Conda:
   ```bash
   conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
   ```

5. Set LD_LIBRARY_PATH environment variable:
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   ```

6. Logout and run `bsub` again to request a node with GPU:
   ```bash
   logout
   bsub -q gpu-short -gpu num=1:j_exclusive=yes:gmem=8G -Is /bin/bash
   ```

7. Activate the "tf" environment again:
   ```bash
   conda activate tf
   ```

8. Install TensorFlow version 2.10 using pip:
   ```bash
   ml miniconda
   conda activate tf
   python3 -m pip install tensorflow==2.10
   ```

## Test TensorFlow Installation:

Check available GPU:
```bash
python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

Expected output:
```
Num GPUs Available:  1
```

```

This Markdown file outlines the steps to install TensorFlow with GPU support using Miniconda on your cluster. Make sure to replace `<your_cuda_version>` with the appropriate CUDA version mentioned in your cluster's documentation. Additionally, modify the TensorFlow version according to your requirements if necessary.

```
ml miniconda && conda create --name=ml-gpu python=3.9 && conda activate ml-gpu && conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0 && mkdir -p $CONDA_PREFIX/etc/conda/activate.d && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && logout && bsub -q gpu-short -gpu num=1:j_exclusive=yes:gmem=8G -Is /bin/bash && conda activate ml-gpu && ml miniconda && conda activate ml-gpu && python3 -m pip install tensorflow==2.10
```