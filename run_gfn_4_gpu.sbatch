#!/bin/bash
#SBATCH --job-name=gfn-DDP-4GPU-PROD
#SBATCH --output=gfn_ddp_4gpu_prod_%j.out
#SBATCH --error=gfn_ddp_4gpu_prod_%j.err
#SBATCH --partition=i64m1tga40u    # <--- Target a healthy, multi-GPU partition
#SBATCH --time=00:29:00            # Adjust time as needed for longer runs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # <--- KEY CHANGE: Request 4 tasks/processes
#SBATCH --gres=gpu:a40:4             # <--- KEY CHANGE: Request 4 'a40' GPUs
#SBATCH --cpus-per-task=8            # Adjust CPUs per task (total 32 CPUs for 4 tasks)

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

# --- DDP Fixes ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# --- Logic to Start or Resume Training ---
CHECKPOINT_FILE="checkpoints/last.ckpt"

# Base command: DDP strategy, 4 devices, and configurable checkpoint interval
# Set checkpoint_save_interval to a reasonable value for actual training (e.g., 200)
BASE_CMD="python train.py trainer.strategy=ddp trainer.devices=4 task.checkpoint_save_interval=200"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found checkpoint file: $CHECKPOINT_FILE. Resuming training. ---"
    CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No checkpoint file found. Starting a new training run. ---"
    CMD="$BASE_CMD"
fi

# --- Run Command with Manual GPU Assignment (Scales automatically) ---
# srun will now launch 4 bash shells. Each will get its SLURM_LOCALID (0,1,2,3)
# and assign its respective GPU.
srun bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID &&
  echo "Node: $SLURMD_NODENAME, Rank $SLURM_PROCID (local rank $SLURM_LOCALID) assigned to CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" &&
  '"$CMD"'
'
    
echo "DDP 4-GPU production job finished."