#!/bin/bash

# ==============================================================================
# Master Experiment Script for GFlowNet Hyperparameter Tuning (10 Epochs)
# ==============================================================================

# --- Base Configuration ---
PYTHON_EXECUTABLE="/home/zeshenghong/miniconda3/envs/gfn_stable/bin/python"
echo "--- Using Python from: $PYTHON_EXECUTABLE ---"

# --- Experiment Definitions ---
EXPERIMENTS=$(cat <<-END
0.01,2.0,0.5
0.05,2.5,0.5
0.1,2.5,0.25
0.2,1.5,0.5
END
)

# --- Base command for launching training ---
# --- MODIFIED: Changed max_epochs to 10 ---
BASE_LAUNCH_CMD="$PYTHON_EXECUTABLE -m torch.distributed.run --nproc_per_node=4 --master_port=29501 train.py \
    trainer.strategy=ddp \
    trainer.devices=4 \
    +trainer.max_epochs=10" # <-- Change this from 2 to 10

# --- Logging and Results ---
RESULTS_LOG="experiment_results.log"
> "$RESULTS_LOG"

# --- MODIFIED: Update the descriptive message for 10 epochs ---
echo "--- Starting Hyperparameter Sweep (10 Epochs per run) ---"
echo "Results will be logged to: $RESULTS_LOG"
echo "-------------------------------------------------------------"

# ==============================================================================
# Main Experiment Loop
# ==============================================================================
experiment_num=1
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    IFS=',' read -r likelihood_weight pf_temp_high use_buffer_prob <<< "$line"

    RUN_NAME="exp${experiment_num}_lw${likelihood_weight}_pt${pf_temp_high}_ubp${use_buffer_prob}"
    CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
    LOG_FILE="training_logs/${RUN_NAME}.log"

    echo ""
    echo "======================================================================"
    echo "Starting Experiment #$experiment_num: $RUN_NAME"
    echo "  - Likelihood Weight: $likelihood_weight"
    echo "  - PF Temp High:      $pf_temp_high"
    echo "  - Use Buffer Prob:   $use_buffer_prob"
    # --- MODIFIED: Update the echoed Max Epochs for clarity ---
    echo "  - Max Epochs:        10" 
    echo "  - Checkpoint Dir:    $CHECKPOINT_DIR"
    echo "  - Log File:          $LOG_FILE"
    echo "======================================================================"

    mkdir -p checkpoints
    mkdir -p training_logs

    CALLBACK_CONFIG_YAML="
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: \"$CHECKPOINT_DIR\"
  save_last: true
  monitor: \"train/loss_step\"
  mode: \"min\"
  save_top_k: 1
  every_n_train_steps: 20 # Still saves frequently within these 10 epochs
  save_on_train_epoch_end: false
  filename: \"best-loss-step={step}\"
"
    TEMP_CALLBACK_FILE="configs/callbacks/temp_exp_callback.yaml"
    echo "$CALLBACK_CONFIG_YAML" > "$TEMP_CALLBACK_FILE"

    FULL_CMD="$BASE_LAUNCH_CMD \
        callbacks=temp_exp_callback \
        task.likelihood_weight=$likelihood_weight \
        task.pf_temp_high=$pf_temp_high \
        task.use_buffer_prob=$use_buffer_prob \
        logger.name=$RUN_NAME \
        logger.group='Hyperparam_Sweep_10_Epochs'" # Maybe a new group name for clarity

    echo "Executing: $FULL_CMD"
    $FULL_CMD > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Experiment #$experiment_num ($RUN_NAME) completed successfully."
        WANDB_DIR=$(grep -o 'wandb/offline-run-[^ ]*' "$LOG_FILE" | head -1)
        echo "$experiment_num,$RUN_NAME,$likelihood_weight,$pf_temp_high,$use_buffer_prob,$WANDB_DIR" >> "$RESULTS_LOG"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "Experiment #$experiment_num ($RUN_NAME) FAILED. Check log: $LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$experiment_num,$RUN_NAME,$likelihood_weight,$pf_temp_high,$use_buffer_prob,FAILED" >> "$RESULTS_LOG"
    fi

    experiment_num=$((experiment_num + 1))

done <<< "$EXPERIMENTS"

rm "$TEMP_CALLBACK_FILE"

echo ""
echo "======================================================================"
echo "All experiments finished."
echo "======================================================================"
echo ""

# ==============================================================================
# Generate Summary Table
# ==============================================================================
echo "Generating summary tables..."

MARKDOWN_FILE="experiment_summary.md"
echo "# Experiment Summary (10-Epoch Run)" > "$MARKDOWN_FILE"
echo "" >> "$MARKDOWN_FILE"
echo "| Exp | Likelihood Weight | PF Temp High | Use Buffer Prob | W&B Log Directory |" >> "$MARKDOWN_FILE"
echo "|:---:|:-----------------:|:------------:|:---------------:|:------------------|" >> "$MARKDOWN_FILE"
while IFS=, read -r exp_num run_name lw pft ubp wb_dir; do
    echo "| $exp_num | \`$lw\` | \`$pft\` | \`$ubp\` | \`$wb_dir\` |" >> "$MARKDOWN_FILE"
done < "$RESULTS_LOG"

LATEX_FILE="experiment_summary.tex"
echo "\\begin{table}[h!]" > "$LATEX_FILE"
echo "  \\centering" >> "$LATEX_FILE"
echo "  \\caption{Hyperparameter Experiment Results (10-Epoch Run)}" >> "$LATEX_FILE"
echo "  \\label{tab:exp_results_10epochs}" >> "$LATEX_FILE"
echo "  \\begin{tabular}{rcccl}" >> "$LATEX_FILE"
echo "    \\toprule" >> "$LATEX_FILE"
echo "    \\textbf{Exp} & \\textbf{Likelihood Wt.} & \\textbf{PF Temp High} & \\textbf{Buffer Prob} & \\textbf{W\&B Log Directory} \\\\" >> "$LATEX_FILE"
echo "    \\midrule" >> "$LATEX_FILE"
while IFS=, read -r exp_num run_name lw pft ubp wb_dir; do
    wb_dir_tex="${wb_dir//_/\\_}"
    echo "    $exp_num & $lw & $pft & $ubp & \\texttt{$wb_dir_tex} \\\\" >> "$LATEX_FILE"
done < "$RESULTS_LOG"
echo "    \\bottomrule" >> "$LATEX_FILE"
echo "  \\end{tabular}" >> "$LATEX_FILE"
echo "\\end{table}" >> "$LATEX_FILE"

echo "Summary tables created: $MARKDOWN_FILE and $LATEX_FILE"
echo "To sync all offline runs to the cloud, run: wandb sync --sync-all"