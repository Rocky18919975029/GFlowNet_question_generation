#!/bin/bash

# --- Environment Setup (Hard-coded Python Path) ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"
echo "--- Assuming Conda environment: /home/zeshenghong/miniconda3/envs/gfn_stable ---"

# --- Redis Server Logic ---
if ! pgrep -x "redis-server" > /dev/null; then
    echo "--- Starting Redis server ---"
    REDIS_EXE=$(which redis-server)
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
# --- FIX: Disable nested tqdm progress bars to clean up the log ---
export TQDM_DISABLE=True

# --- Hydra Overrides for the Diagnostic Run ---
BASE_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    callbacks=diag_metrics_checkpoint \
    +trainer.val_check_interval=2"

# --- Point to the NEW checkpoint directory for resumption logic ---
CHECKPOINT_FILE="diag_metrics_checkpoints/last.ckpt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found diagnostic checkpoint: $CHECKPOINT_FILE. Resuming diagnostic run. ---"
    FULL_CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No diagnostic checkpoint found in '$CHECKPOINT_FILE'. Starting a new diagnostic run. ---"
    FULL_CMD="$BASE_CMD"
fi

# --- Execute the Training Command ---
echo "--- Running command: $FULL_CMD ---"
# --- Log to a separate file for the diagnostic run ---
nohup $FULL_CMD > training_diag_metrics.log 2>&1 &

echo "Diagnostic training job submitted to background. Monitor with 'tail -f training_diag_metrics.log' and 'nvidia-smi'."