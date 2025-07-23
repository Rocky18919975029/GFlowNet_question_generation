#!/bin/bash

# ==============================================================================
# Experiment Script for Answer Quality Reward Hyperparameter Tuning
# (Flexible Version)
# ==============================================================================
# This script runs a sweep to find the optimal `answer_quality_weight`.
# It can be run "as-is" for the full 20-epoch sweep, or it can be run
# with additional command-line arguments to override settings for a fast
# smoke test (e.g., `./script.sh trainer.max_steps=15`).
#
# It assumes the replay buffer has been pre-seeded.
# ==============================================================================

export WANDB_DIR=$(pwd)/wandb

# --- Base Configuration ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# ==============================================================================
# Pre-flight Check for Seed Data
# ==============================================================================
echo ""
echo "--- Performing pre-flight check for seeded data in Redis... ---"
if ! command -v redis-cli &> /dev/null
then
    echo "WARNING: 'redis-cli' command not found. Cannot verify if the buffer is seeded. Proceeding with caution."
else
    KEY_COUNT=$(redis-cli -t 1 DBSIZE 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! CRITICAL ERROR: Could not connect to Redis server via redis-cli.   !!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi

    if [ "$KEY_COUNT" -lt 100 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! CRITICAL ERROR: Redis database appears to be empty or not seeded. !!"
        echo "!! Found only $KEY_COUNT keys. Expected several hundred.                !!"
        echo "!! Please run the re-seeding script successfully before this.         !!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    else
        echo "--- Pre-flight check PASSED. Found $KEY_COUNT keys in Redis. Proceeding... ---"
    fi
fi
# ==============================================================================

# --- Experiment Definitions (for the full sweep) ---
# This list is used for the full sweep.
# For a smoke test, override this from the command line by passing in a single
# value, e.g., 'task.answer_quality_weight=0.1'
EXPERIMENTS=$(cat <<-END
0.1
0.25
0.5
1.0
END
)

# --- Base command for launching training ---
BASE_LAUNCH_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29502 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    trainer.max_epochs=20 \
    training.log_every_n_steps=10 \
    logger.group='AnswerQuality_Sweep'"

# --- Logging and Results ---
RESULTS_LOG="answer_quality_sweep_results.log"
> "$RESULTS_LOG"
echo "--- Starting Answer Quality Weight Hyperparameter Sweep ---"
echo "--- NOTE: This sweep uses the pre-seeded replay buffer. ---"
echo "--- Final checkpoint of each run will be saved for potential resumption ---"

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
experiment_num=1
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    answer_quality_weight="$line"

    RUN_NAME="aqw_sweep_exp${experiment_num}_aqw${answer_quality_weight}"
    CHECKPOINT_DIR="${EXTERNAL_STORAGE_PATH}/checkpoints/${RUN_NAME}"
    LOG_FILE="training_logs/${RUN_NAME}.log"

    echo ""
    echo "======================================================================"
    echo "Starting Experiment #$experiment_num: $RUN_NAME"
    echo "  - Answer Quality Weight: $answer_quality_weight"
    echo "  - Checkpoint Dir:        $CHECKPOINT_DIR"
    echo "======================================================================"

    mkdir -p "${EXTERNAL_STORAGE_PATH}/checkpoints"
    mkdir -p training_logs

    # Dynamically create the callback config for this specific run.
    CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/loss_step\"
  mode: \"min\"
  save_top_k: 1
  every_n_train_steps: 1000
  save_on_train_epoch_end: false
  filename: \"best-loss-step={step}\"
"
    TEMP_CALLBACK_FILE="configs/callbacks/temp_aqw_sweep_callback.yaml"
    echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

    # Assemble the full command, adding the specific parameters for this run
    # and forwarding any additional arguments passed to this script via `$@`.
    FULL_CMD="$BASE_LAUNCH_CMD \
        callbacks=temp_aqw_sweep_callback \
        task.answer_quality_weight=$answer_quality_weight \
        task.answer_failure_penalty=0.0 \
        logger.name=$RUN_NAME \
        $@"

    echo "Executing: $FULL_CMD"
    $FULL_CMD > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Experiment #$experiment_num ($RUN_NAME) completed successfully."
        WANDB_DIR=$(grep -o 'wandb/offline-run-[^ ]*' "$LOG_FILE" | head -1)
        echo "$experiment_num,$RUN_NAME,$answer_quality_weight,$WANDB_DIR" >> "$RESULTS_LOG"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Experiment #$experiment_num ($RUN_NAME) FAILED. Check log: $LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$experiment_num,$RUN_NAME,$answer_quality_weight,FAILED" >> "$RESULTS_LOG"
    fi

    experiment_num=$((experiment_num + 1))

done <<< "$EXPERIMENTS"

rm "$TEMP_CALLBACK_FILE"

echo ""
echo "======================================================================"
echo "All answer quality weight experiments finished."
echo "Each run's final state is saved in its respective checkpoint directory."
echo "======================================================================"