#!/bin/bash

# ==============================================================================
# Diagnostic Script for Hybrid Reward GFlowNet Training (4-GPU DDP)
# ==============================================================================
# This script launches a single, persistent training run using the new
# hybrid reward strategy.
#
# *** NOTE: Checkpoints are saved to a large-capacity directory. ***
# ==============================================================================

# --- Environment Setup ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- CHANGE: Updated the external storage path ---
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Redis Server Logic ---
if ! pgrep -x "redis-server" > /dev/null; then
    echo "--- Starting Redis server on port 6379 ---"
    nohup redis-server --port 6379 > redis_hybrid.log 2>&1 &
    sleep 5
    echo "--- Redis server started ---"
else
    echo "--- Redis server already running ---"
fi

# --- DDP Environment Variables ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TQDM_DISABLE=True

# --- Hydra Overrides for the Hybrid Reward Diagnostic Run ---
BASE_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    +trainer.max_epochs=10 \
    training.log_every_n_steps=10 \
    task.use_hybrid_reward=true \
    task.penalized_reward_end_step=150 \
    logger.group='HybridReward_Diagnostic_10Epoch'"

# --- Checkpoint Configuration ---
HYBRID_CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/hybrid_checkpoints/"
mkdir -p "$HYBRID_CHECKPOINT_DIR"

# Dynamically create the callback config for this run
CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$HYBRID_CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/loss_step\"
  mode: \"min\"
  save_top_k: 1
  every_n_train_steps: 20
  save_on_train_epoch_end: false
  filename: \"best-loss-step={step}\"
"
TEMP_CALLBACK_FILE="configs/callbacks/temp_diag_callback.yaml"
echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

# Append the dynamic callback to the command
BASE_CMD="$BASE_CMD callbacks=temp_diag_callback"

# --- Resumption Logic ---
CHECKPOINT_FILE="${HYBRID_CHECKPOINT_DIR}last.ckpt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found hybrid checkpoint: $CHECKPOINT_FILE. Resuming run. ---"
    FULL_CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No checkpoint found in '$CHECKPOINT_FILE'. Starting new hybrid run. ---"
    FULL_CMD="$BASE_CMD"
fi

# --- Execute the Training Command ---
LOG_FILE="training_hybrid_diag.log"
echo "--- Running command: $FULL_CMD ---"
echo "--- Logging output to: $LOG_FILE ---"

nohup $FULL_CMD > "$LOG_FILE" 2>&1 &

echo ""
echo "Hybrid reward diagnostic training job submitted to background."
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  watch nvidia-smi"
echo ""
echo "Checkpoints are now in '$HYBRID_CHECKPOINT_DIR'."