#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=bdau-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1                     
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=my_files/logs/run_%j.out
#SBATCH --error=my_files/logs/run_%j.err


srun python test_neural_controller.py