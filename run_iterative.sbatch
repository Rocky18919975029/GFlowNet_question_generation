#!/bin/bash
#SBATCH --job-name=gfn-S-GPU-ITERATIVE-FINAL
#SBATCH --output=gfn_s_gpu_iterative_final_%j.out
#SBATCH --error=gfn_s_gpu_iterative_final_%j.err
#SBATCH --partition=debug
#SBATCH --nodelist=gpu3-9
#SBATCH --time=00:29:00
#SBATCH --nodes=1             # Request only ONE node
#SBATCH --ntasks-per-node=1   # Request only ONE task/process on that node
#SBATCH --gres=gpu:1          # Request only ONE GPU for this task
#SBATCH --cpus-per-task=8

# --- Environment Setup ---
module purge
module load slurm
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gfn_stable # Use your stable environment

# --- Start Redis Server ---
REDIS_EXE="$(conda info --base)/envs/gfn_stable/bin/redis-server"
echo "--- Starting Redis server ---"
$REDIS_EXE --daemonize yes --port 6379
sleep 5

# --- DDP Fixes (Harmless in single-GPU, good for consistency) ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# --- Logic to Start or Resume Training ---
CHECKPOINT_FILE="checkpoints/last.ckpt"

# BASE_CMD now explicitly sets trainer.devices=1 for single-GPU mode.
# We no longer specify any callbacks, relying on the manual save in lightning_module.py.
# Use a small checkpoint_save_interval for testing (e.g., 20)
BASE_CMD="python train.py trainer.strategy=auto trainer.devices=1 task.checkpoint_save_interval=20"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found checkpoint file: $CHECKPOINT_FILE. Resuming training. ---"
    CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No checkpoint file found. Starting a new training run. ---"
    CMD="$BASE_CMD"
fi

# --- Run Command on GPU 0 ---
# This ensures only GPU 0 is used by the single task.
srun bash -c '
  export CUDA_VISIBLE_DEVICES=0 && # Force usage of GPU 0
  echo "Node: $SLURMD_NODENAME, Rank 0 (forced) assigned to CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" &&
  '"$CMD"'
'
    
echo "Single-GPU iterative job finished."