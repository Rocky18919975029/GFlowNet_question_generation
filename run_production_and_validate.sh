#!/bin/bash

# ==============================================================================
# Production & Validation Script for Hybrid Reward GFlowNet
# ==============================================================================
# This script launches a single, long-running production job using the winning
# hyperparameters from Experiment 3, but with a new random seed for validation.
#
# How to Use:
# 1. Let the job run.
# 2. After ~10 epochs, compare its W&B charts to the original Exp 3 run.
# 3. If the initial learning dynamics are similar, the strategy is validated,
#    and this run can continue to completion to become the final production model.
# ==============================================================================

# --- Environment Setup ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- External Storage Path ---
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Redis Server Logic ---
if ! pgrep -x "redis-server" > /dev/null; then
    echo "--- Starting Redis server on port 6379 ---"
    nohup redis-server --port 6379 > redis_production.log 2>&1 &
    sleep 5
    echo "--- Redis server started ---"
else
    echo "--- Redis server already running ---"
fi

# --- DDP Environment Variables ---
export TOKENIZERS_PARALLETISM=false
export OMP_NUM_THREADS=1
export TQDM_DISABLE=True

# --- Hydra Overrides for the Production & Validation Run ---
BASE_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    trainer.max_epochs=50 \
    training.log_every_n_steps=10 \
    task.use_hybrid_reward=true \
    task.penalized_reward_end_step=150 \
    task.contradiction_threshold=-6.0 \
    seed=43 \
    logger.group='Production_Runs'"

# --- Checkpoint Configuration ---
PRODUCTION_CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/production_checkpoints_seed43/"
mkdir -p "$PRODUCTION_CHECKPOINT_DIR"

# Dynamically create the callback config for this run
CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$PRODUCTION_CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/avg_log_reward_unscaled\" # Monitor a more meaningful metric for the best model
  mode: \"max\"
  save_top_k: 3 # Save the top 3 best models over the long run
  every_n_train_steps: 100 # Save less frequently for a long run
  save_on_train_epoch_end: false
  filename: \"best-reward-step={step}\"
"
TEMP_CALLBACK_FILE="configs/callbacks/temp_production_callback.yaml"
echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

# Append the dynamic callback to the command
BASE_CMD="$BASE_CMD callbacks=temp_production_callback"

# --- Resumption Logic ---
CHECKPOINT_FILE="${PRODUCTION_CHECKPOINT_DIR}last.ckpt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "--- Found production checkpoint: $CHECKPOINT_FILE. Resuming run. ---"
    FULL_CMD="$BASE_CMD +trainer.resume_from_checkpoint=$CHECKPOINT_FILE"
else
    echo "--- No checkpoint found. Starting new production & validation run. ---"
    FULL_CMD="$BASE_CMD"
fi

# --- Execute the Training Command ---
LOG_FILE="training_production_seed43.log"
echo "--- Running command: $FULL_CMD ---"
echo "--- Logging output to: $LOG_FILE ---"

nohup $FULL_CMD > "$LOG_FILE" 2>&1 &

echo ""
echo "Production & Validation job submitted to background."
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  watch nvidia-smi"
echo ""
echo "Checkpoints will be saved in '$PRODUCTION_CHECKPOINT_DIR'."