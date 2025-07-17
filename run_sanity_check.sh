#!/bin/bash
# ==============================================================================
# The Simplest Possible Sanity Check Script
# ==============================================================================

PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
LOG_FILE="training_logs/final_sanity_check.log"
RUN_NAME="final_sanity_check"
CHECKPOINT_DIR="checkpoints/final_sanity_check"

# --- Create necessary directories ---
mkdir -p checkpoints
mkdir -p training_logs

# --- Create the callback config ---
# We still need this to save checkpoints to a unique directory.
TEMP_CALLBACK_FILE="configs/callbacks/temp_sanity_check_callback.yaml"
cat > "$TEMP_CALLBACK_FILE" <<'EOF'
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: "placeholder"
  save_last: true
EOF

echo "--- Starting Final Sanity Check ---"
echo "--- Running 1 epoch on 16 samples ---"
echo "-------------------------------------"

# --- The Command ---
# It uses the tiny_test_config and overrides the callback and logger names.
$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py \
    --config-name=tiny_test_config \
    callbacks=temp_sanity_check_callback \
    callbacks.0.dirpath=$CHECKPOINT_DIR \
    logger.name=$RUN_NAME \
    logger.group='Final_Sanity_Check' \
    > "$LOG_FILE" 2>&1

# --- Check for Success ---
if [ $? -eq 0 ]; then
    echo "Sanity check completed successfully."
    WANDB_DIR=$(grep -o 'wandb/offline-run-[^ ]*' "$LOG_FILE" | head -1)
    echo "W&B log directory: $WANDB_DIR"
    echo "To sync, run: wandb sync $WANDB_DIR"
else
    echo "Sanity check FAILED. Check log: $LOG_FILE"
fi

# Clean up the temporary file
rm "$TEMP_CALLBACK_FILE"