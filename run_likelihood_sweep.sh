#!/bin/bash

# ==============================================================================
# Experiment Script for Likelihood Weight Hyperparameter Tuning
# ==============================================================================
# This script runs a sweep to find the optimal `likelihood_weight` for the
# question's fluency score, using the best hybrid reward settings.
#
# Each experiment is run for 10 epochs and its final checkpoint ('last.ckpt')
# is saved to a unique directory. This allows any promising run to be
# resumed for a longer training duration later.
# ==============================================================================

# --- Base Configuration ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- Experiment Definitions (likelihood_weight) ---
# Testing a range from moderately higher to much higher than the default 0.2
EXPERIMENTS=$(cat <<-END
0.5
1.0
2.0
5.0
END
)

# --- Base command for launching training ---
# Uses the new defaults from config.yaml (threshold=-6.0, end_step=150)
BASE_LAUNCH_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    trainer.max_epochs=20 \
    training.log_every_n_steps=10 \
    logger.group='LikelihoodWeight_Sweep'"

# --- Logging and Results ---
RESULTS_LOG="likelihood_sweep_results.log"
> "$RESULTS_LOG"
echo "--- Starting Likelihood Weight Hyperparameter Sweep (10 Epochs per run) ---"
echo "--- Final checkpoint of each run will be saved for potential resumption ---"

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
experiment_num=1
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    likelihood_weight="$line"

    RUN_NAME="lw_sweep_exp${experiment_num}_lw${likelihood_weight}"
    CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/checkpoints/${RUN_NAME}"
    LOG_FILE="training_logs/${RUN_NAME}.log"

    echo ""
    echo "======================================================================"
    echo "Starting Experiment #$experiment_num: $RUN_NAME"
    echo "  - Likelihood Weight: $likelihood_weight"
    echo "  - Checkpoint Dir:    $CHECKPOINT_DIR"
    echo "======================================================================"

    mkdir -p "${EXTERNAL_STORAGE_PATH}/checkpoints"
    mkdir -p training_logs

    # Dynamically create the callback config for this specific run.
    # `save_last: true` is crucial for easy resumption.
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
    TEMP_CALLBACK_FILE="configs/callbacks/temp_lw_sweep_callback.yaml"
    echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

    FULL_CMD="$BASE_LAUNCH_CMD \
        callbacks=temp_lw_sweep_callback \
        task.likelihood_weight=$likelihood_weight \
        logger.name=$RUN_NAME"

    echo "Executing: $FULL_CMD"
    $FULL_CMD > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Experiment #$experiment_num ($RUN_NAME) completed successfully."
        WANDB_DIR=$(grep -o 'wandb/offline-run-[^ ]*' "$LOG_FILE" | head -1)
        echo "$experiment_num,$RUN_NAME,$likelihood_weight,$WANDB_DIR" >> "$RESULTS_LOG"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Experiment #$experiment_num ($RUN_NAME) FAILED. Check log: $LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$experiment_num,$RUN_NAME,$likelihood_weight,FAILED" >> "$RESULTS_LOG"
    fi

    experiment_num=$((experiment_num + 1))

done <<< "$EXPERIMENTS"

rm "$TEMP_CALLBACK_FILE"

echo ""
echo "======================================================================"
echo "All likelihood weight experiments finished."
echo "Each run's final state is saved in its respective checkpoint directory."
echo "Example: to resume Exp #3 for 50 total epochs, run a command like:"
echo "python train.py ... +trainer.max_epochs=50 +trainer.resume_from_checkpoint=${EXTERNAL_STORAGE_PATH}/checkpoints/lw_sweep_exp3_lw2.0/last.ckpt"
echo "======================================================================"

# --- (Summary table generation would be added here, similar to other scripts) ---