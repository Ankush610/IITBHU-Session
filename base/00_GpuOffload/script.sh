#!/bin/bash
#SBATCH --job-name=mnist_code           # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Number of tasks (one per GPU per node)
#SBATCH --gres=gpu:1                   # Number of GPUs on each node
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --partition=gpu                # GPU partition
#SBATCH --output=logs_%j.out           # Output log file
#SBATCH --error=logs_%j.err            # Error log file
#SBATCH --time=02:00:00                # Time limit
#SBATCH --qos=gpumultinode
#SBATCH --reservation=hpcws

source /scratch/hpcws.2/setup/working/dependencies.sh 

# Run the PyTorch script with torchrun
echo "===================================="
echo "Starting Training with torch-GPU/CPU"
echo "===================================="

srun python mnist_code_cpu_gpu.py

# Log the completion of the job
echo "===================================="
echo "Training Completed"
echo "===================================="


