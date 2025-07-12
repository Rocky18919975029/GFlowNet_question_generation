#!/bin/bash
#SBATCH --job-name=gfn-debug-test
#SBATCH --output=gfn_debug_%j.out
#SBATCH --error=gfn_debug_%j.err
#SBATCH --partition=debug
#SBATCH --nodes=1
# --- MODIFICATION 1: Request all GPUs on the node in one go ---
# Instead of tasks per GPU, just request the total number of GPUs for the job.
# Let's request 2 GPUs on this single node.
#SBATCH --gpus=2
# --- MODIFICATION 2: Launch only ONE task per node ---
# This task will be the main process that Lightning uses to launch workers.
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 # Give the main task enough CPUs for its workers
#SBATCH --time=00:29:00

module purge
module load slurm

# Environment variables are no longer needed as Lightning will set them for its child processes
# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=29500

# --- MODIFICATION 3: We don't need the wrapper or srun anymore ---
# We just run the python command directly.
# Lightning will handle the distributed launch.

# First, initialize conda for this shell
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc2hdd/home/zhongal/miniconda3/envs/gfn

# Now, run the python script.
# We tell the trainer it has 2 devices available.
python train.py \
    trainer.strategy=ddp \
    trainer.devices=2 \
    trainer.num_nodes=1

echo "Debug job finished."