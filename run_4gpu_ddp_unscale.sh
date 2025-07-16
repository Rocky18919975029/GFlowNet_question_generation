#!/bin/bash

# --- Environment Setup (Hard-coded Python Path) ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $(which "$PYTHON_EXECUTABLE") ---"
echo "--- Assuming Conda environment: /home/zeshenghong/miniconda3/envs/gfn_stable ---"

# ... (Redis server logic is unchanged) ...
if ! pgrep -x "redis-server" > /dev/null; then
    echo "--- Starting Redis server ---"
    REDIS_EXE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/redis-server"
    if [ -z "$REDIS_EXE" ] || [ ! -x "$REDIS_EXE" ]; then
        echo "ERROR: redis-server executable not found or not executable. Please verify its path."
        exit 1
    fi
    nohup "$REDIS_EXE" --port 6379 > redis.log 2>&1 &
    sleep 5
    echo "--- Redis server started ---"
else
    echo "--- Redis server already running ---"
fi

# --- DDP Environment Variables ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# --- Logic to Start or Resume Training from the NEW directory ---
CHECKPOINT_FILE="production_checkpoints/last.ckpt" # <-- UPDATE THIS PATH

# --- MODIFIED: Use the new 'production_checkpoint' configuration ---
BASE_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    callbacks=production_checkpoint \
    +trainer.val_check_interval=20" # Match the new every_n_train_steps

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found checkpoint file: $CHECKPOINT_FILE. Resuming training. ---"
    FULL_CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No checkpoint file found in 'production_checkpoints/'. Starting a new training run. ---"
    FULL_CMD="$BASE_CMD"
fi

# --- Execute the Training Command ---
echo "--- Running command: $FULL_CMD ---"
nohup $FULL_CMD > training_production.log 2>&1 & # <-- Changed log file name

echo "Training job submitted to background. Monitor with 'tail -f training_production.log' and 'nvidia-smi'."