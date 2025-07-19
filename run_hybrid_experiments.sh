#!/bin/bash

# ==============================================================================
# Experiment Script for Hybrid Reward GFlowNet Hyperparameter Tuning
# ==============================================================================
# This script runs a sweep of experiments to test the new hybrid reward
# parameters.
#
# *** NOTE: Checkpoints are saved to a large-capacity directory. ***
# ==============================================================================

# --- Base Configuration ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- CHANGE: Updated the external storage path ---
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Experiment Definitions (contradiction_threshold) ---
EXPERIMENTS=$(cat <<-END
-4.0
-5.0
-6.0
-7.0
END
)

# --- Base command for launching training ---
BASE_LAUNCH_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    +trainer.max_epochs=20 \
    training.log_every_n_steps=10 \
    task.use_hybrid_reward=true \
    task.penalized_reward_end_step=150"

# --- Logging and Results ---
RESULTS_LOG="hybrid_experiment_results.log"
> "$RESULTS_LOG"

echo "--- Starting Hybrid Reward Hyperparameter Sweep (10 Epochs per run) ---"
echo "Results will be logged to: $RESULTS_LOG"
echo "---------------------------------------------------------------------"

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
experiment_num=1
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    threshold="$line"

    RUN_NAME="hybrid_exp${experiment_num}_thresh${threshold}"
    CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/checkpoints/${RUN_NAME}"
    LOG_FILE="training_logs/${RUN_NAME}.log"

    echo ""
    echo "======================================================================"
    echo "Starting Experiment #$experiment_num: $RUN_NAME"
    echo "  - Contradiction Threshold:   $threshold"
    echo "  - Checkpoint Dir:            $CHECKPOINT_DIR"
    echo "  - Log File:                  $LOG_FILE"
    echo "======================================================================"

    mkdir -p "${EXTERNAL_STORAGE_PATH}/checkpoints"
    mkdir -p training_logs

    # Dynamically create the callback config for this specific run
    CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/loss_step\"
  mode: \"min\"
  save_top_k: 1
  every_n_train_steps: 20
  save_on_train_epoch_end: false
  filename: \"best-loss-step={step}\"
"
    TEMP_CALLBACK_FILE="configs/callbacks/temp_hybrid_exp_callback.yaml"
    echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

    FULL_CMD="$BASE_LAUNCH_CMD \
        callbacks=temp_hybrid_exp_callback \
        task.contradiction_threshold=$threshold \
        logger.name=$RUN_NAME \
        logger.group='HybridReward_Sweep_Thresh'"

    echo "Executing: $FULL_CMD"
    $FULL_CMD > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Experiment #$experiment_num ($RUN_NAME) completed successfully."
        WANDB_DIR=$(grep -o 'wandb/offline-run-[^ ]*' "$LOG_FILE" | head -1)
        echo "$experiment_num,$RUN_NAME,$threshold,$WANDB_DIR" >> "$RESULTS_LOG"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Experiment #$experiment_num ($RUN_NAME) FAILED. Check log: $LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$experiment_num,$RUN_NAME,$threshold,FAILED" >> "$RESULTS_LOG"
    fi

    experiment_num=$((experiment_num + 1))

done <<< "$EXPERIMENTS"

rm "$TEMP_CALLBACK_FILE"

echo ""
echo "======================================================================"
echo "All hybrid experiments finished."
echo "======================================================================"
echo ""

# --- Summary generation remains the same ---
echo "Generating summary tables..."
MARKDOWN_FILE="hybrid_experiment_summary.md"
echo "# Hybrid Reward Experiment Summary" > "$MARKDOWN_FILE"
echo "" >> "$MARKDOWN_FILE"
echo "| Exp | Contradiction Threshold | W&B Log Directory |" >> "$MARKWORD_FILE"
echo "|:---:|:-----------------------:|:------------------|" >> "$MARKDOWN_FILE"
while IFS=, read -r exp_num run_name ct wb_dir; do
    echo "| $exp_num | \`$ct\` | \`$wb_dir\` |" >> "$MARKDOWN_FILE"
done < "$RESULTS_LOG"

echo "Summary table created: $MARKDOWN_FILE"
echo "To sync all offline runs to the cloud, run: wandb sync --sync-all"