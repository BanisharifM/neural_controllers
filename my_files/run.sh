#!/bin/bash
#SBATCH --job-name=poetry
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=my_files/logs/poetry_%j.out
#SBATCH --error=my_files/logs/poetry_%j.err

# Load environment variables
source /work/hdd/bdau/mbanisharifdehkordi/neural_controllers/.env

# Set wandb mode
export WANDB_MODE=online

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"

# Use the Python directly from your conda environment
/projects/bdau/envs/ncontrollers_env/bin/python my_files/src/test_stronger_poetry.py