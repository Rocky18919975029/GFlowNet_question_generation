#!/bin/bash

# ==============================================================================
# Fast DDP Debugging & Resumption Test Script
# ==============================================================================
# This script launches a full 4-GPU DDP training run but is configured to
# be extremely short, for rapid debugging of distributed logic, race
# conditions, and checkpoint/resume flows in a DDP environment.
#
# *** NOTE: Checkpoints are saved to a large-capacity directory. ***
# ==============================================================================

# --- Environment Setup ---
# IMPORTANT: Adjust this path if your Conda environment's Python executable is different
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- External Storage Path ---
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Redis Server Logic ---
if ! pgrep -x "redis-server" > /dev/null; then
    echo "--- Starting Redis server on port 6379 ---"
    nohup redis-server --port 6379 > redis_ddp_debug.log 2>&1 &
    sleep 2
    echo "--- Redis server started ---"
else
    echo "--- Redis server already running ---"
fi

# --- Environment Variables ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TQDM_DISABLE=True

# --- Hydra Overrides for the Fast DDP Debug Run ---
BASE_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29503 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    +trainer.max_steps=10 \
    training.log_every_n_steps=1 \
    data.limit_data=8 \
    training.accumulate_grad_batches=1 \
    task.use_hybrid_reward=true \
    task.penalized_reward_end_step=5 \
    logger.group='DDP_DEBUG_Runs'"

# --- Checkpoint Configuration ---
DDP_DEBUG_CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/ddp_debug_checkpoints/"
mkdir -p "$DDP_DEBUG_CHECKPOINT_DIR" # Ensure the external directory exists

# Dynamically create the callback config for this run
CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$DDP_DEBUG_CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/loss_step\"
  mode: \"min\"
  save_top_k: 1
  every_n_train_steps: 2
  save_on_train_epoch_end: false
  filename: \"ddp-debug-step={step}\"
"
TEMP_CALLBACK_FILE="configs/callbacks/temp_ddp_debug_callback.yaml"
echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

# Append the dynamic callback to the command
BASE_CMD="$BASE_CMD callbacks=temp_ddp_debug_callback"

# --- Resumption Logic ---
CHECKPOINT_FILE="${DDP_DEBUG_CHECKPOINT_DIR}last.ckpt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "!!! Found DDP debug checkpoint: $CHECKPOINT_FILE. RESUMING a short DDP run. !!!"
    echo ""
    FULL_CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo ""
    echo "--- No DDP debug checkpoint found. Starting a NEW short DDP run. ---"
    echo ""
    FULL_CMD="$BASE_CMD"
fi

# --- Execute the Training Command ---
LOG_FILE="training_ddp_debug.log"
echo "--- Running command: $FULL_CMD ---"
echo "--- Logging output to: $LOG_FILE ---"

# Run in the background and log to a file
nohup $FULL_CMD > "$LOG_FILE" 2>&1 &

echo ""
echo "DDP debug job submitted to background."
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  watch nvidia-smi"
echo ""
echo "Checkpoints are now in '$DDP_DEBUG_CHECKPOINT_DIR'."
echo "To start fresh, delete the '$DDP_DEBUG_CHECKPOINT_DIR' directory."