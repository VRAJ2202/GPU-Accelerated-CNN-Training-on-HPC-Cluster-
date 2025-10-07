#!/bin/bash
#SBATCH --account=tma26892_1425
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=vraj_job.out

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "TMPDIR = $TMPDIR"
echo "=========================================="

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"

cd /home1/vapatel/Concrete_Crack/2

# Load modules
module purge
module load gcc/13.3.0
module load cuda/12.6.3

# Activate conda env
source /home1/vapatel/miniconda3/etc/profile.d/conda.sh
conda activate Py310

# Add system CUDA libraries so TensorFlow can find them
export LD_LIBRARY_PATH=/usr/lib64:/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Optional XLA flag
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Run Python
python Vraj_Patel.py

echo "Program finished with exit code $? at $(date)"