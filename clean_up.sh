# Targeted cleanup for experiment-related files

# 1. Remove the experiment-specific checkpoint parent directory
echo "Removing experiment sweep checkpoint directory..."
rm -rf checkpoints/

# 2. Remove the experiment-specific training log directory
echo "Removing experiment training logs..."
rm -rf training_logs/

# 3. Remove previous experiment summary files
echo "Removing experiment summary files..."
rm -f experiment_summary.md
rm -f experiment_summary.tex
rm -f experiment_results.log

# 4. Remove any temporary callback files
echo "Removing temporary configs..."
rm -f configs/callbacks/temp_exp_callback.yaml

# 5. Remove the failed offline W&B runs to avoid clutter
echo "Removing failed local W&B data..."
rm -rf wandb/offline-run-*

echo ""
echo "Cleanup complete. Ready for a fresh experiment run."