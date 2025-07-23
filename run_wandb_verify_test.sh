#!/bin/bash

# ==============================================================================
# Definitive W&B Table Logging Verification Test
# ==============================================================================
# This script runs a single, short DDP training job that is GUARANTEED to
# be long enough to trigger the wandb.Table logging.
# ==============================================================================

echo "--- Starting Definitive W&B Table Logging Test ---"

# --- Base Configuration ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Experiment Parameters (Hardcoded for the test) ---
RUN_NAME="final_log_verify_$(date +%s)"
CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/checkpoints/${RUN_NAME}"
LOG_FILE="training_logs/${RUN_NAME}.log"

echo ""
echo "======================================================================"
echo "Starting Verification Run: $RUN_NAME"
echo "  - Checkpoint Dir:        $CHECKPOINT_DIR"
echo "======================================================================"

mkdir -p "${EXTERNAL_STORAGE_PATH}/checkpoints"
mkdir -p training_logs

# --- Dynamic Callback Config (no changes needed) ---
CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$CHECKPOINT_DIR\"
  save_last: true
"
TEMP_CALLBACK_FILE="configs/callbacks/temp_verify_test_callback.yaml"
echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"


# ==============================================================================
# The DDP Launch Command with CORRECT parameters for a 10-step run
# ==============================================================================
# Calculation:
# - data.limit_data=320 / 4 GPUs = 80 data points per GPU.
# - 80 data points / accumulate_grad_batches=8 = 10 training steps.
# - trainer.max_epochs=1 ensures it runs for exactly these 10 steps.
# - training.log_every_n_steps=5 ensures logging at steps 5 and 10.
# ==============================================================================
$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    trainer.max_epochs=1 \
    \
    data.limit_data=320 \
    \
    training.log_every_n_steps=5 \
    \
    task.answer_quality_weight=0.1 \
    \
    logger.project="contradiction-gfn-log-test" \
    logger.group="Final-Log-Verification" \
    logger.name=$RUN_NAME \
    logger.mode="offline" \
    \
    callbacks=temp_verify_test_callback \
    > "$LOG_FILE" 2>&1


# --- Cleanup and Final Message ---
rm "$TEMP_CALLBACK_FILE"

if [ $? -eq 0 ]; then
    echo "--- Verification Test Finished Successfully ---"
    echo "A new offline run has been created in the 'wandb/' directory."
    echo "Please sync it by running: wandb sync --sync-all"
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "--- Verification Test FAILED. Check log: $LOG_FILE ---"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi