#!/bin/bash

# Clear the log file right now, before we even talk to SLURM!
> training_log.txt


# 1. Submit the job and capture the output (which contains the Job ID)
SUBMIT_OUTPUT=$(sbatch run_model.slurm)
echo "$SUBMIT_OUTPUT"

# 2. Extract the Job ID number using 'awk'
JOB_ID=$(echo $SUBMIT_OUTPUT | awk '{print $4}')

# 3. Show the current queue status for your user
echo "-----------------------------------"
echo "Current Queue Status:"
squeue -u $USER
echo "-----------------------------------"

# 4. Start streaming the log file
echo "Streaming logs from training_log.txt... (Press Ctrl+C to stop streaming)"
tail -f training_log.txt


# ./launch.sh