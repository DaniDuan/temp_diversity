#!/bin/bash

# =============================================================================
# sim_div.sh
#
# SLURM batch script for running sim_div.jl as a job array.
# Submits 999 independent replicates, each using its SLURM_ARRAY_TASK_ID
# as a random seed, sweeping temperature from 0–30°C.
#
# Submit with:
#   sbatch sim_div.sh
#
# Monitor with:
#   squeue -u $USER
# =============================================================================

#SBATCH --time=0-10:00:00   # Maximum time limit
#SBATCH --ntasks=1          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPU cores per task
#SBATCH --mem=1G            # Memory per node
#SBATCH --partition=large_336
#SBATCH --array=1-999

echo "Julia is about to run"
julia ~/code/Cluster/sim_div.jl
echo "Julia has finished running"

