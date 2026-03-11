#!/bin/bash
#SBATCH --job-name=LFM_Test        # The name of your job 
#SBATCH --account=shocher_prj      # Your lab's account for tracking usage
#SBATCH --partition=l40s-shared    # ⚠️ The queue we found earlier with the L40S GPUs
#SBATCH --gpus=1                   # We are asking for exactly 1 GPU
#SBATCH --cpus-per-task=4          # Asking for 4 CPU cores to load data quickly
#SBATCH --mem=32G                  # Asking for 32GB of RAM
#SBATCH --time=01:00:00            # We are asking for a 4-hour time limit
#SBATCH --output=training_log.txt  # SLURM will save all your print() outputs to this file

# > training_log.txt  # Clear the log file at the start of the job

# 1. Activate the Python environment we built together
source venv/bin/activate  


# This tells Python to look for modules starting from the current directory
export PYTHONPATH=$PYTHONPATH:.


# 2. Run your actual Python code! 
# (Replace 'main.py' with the actual name of the python file you want to run)
python main.py