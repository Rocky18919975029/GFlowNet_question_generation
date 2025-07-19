#!/bin/bash

# ==============================================================================
# Project Cleanup Script
# ==============================================================================
# This script removes generated artifacts like logs, checkpoints, and
# temporary files. It helps ensure a clean state before starting new
# experiment sweeps or formal runs.
#
# Usage:
#   ./clean_up.sh         (Shows usage instructions)
#   ./clean_up.sh experiments  (Removes only experiment/debug artifacts)
#   ./clean_up.sh all          (Removes ALL generated artifacts)
# ==============================================================================

# --- Define the external storage path ---
# This MUST match the path used in your run scripts.
EXTERNAL_STORAGE_PATH="/home/data/zeshenghong/checkpoints/gfn_project"

# --- Function to ask for user confirmation ---
confirm() {
    # call with a prompt string
    read -r -p "$1 [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            true
            ;;
        *)
            false
            ;;
    esac
}

# --- Main Logic ---
if [ "$1" == "experiments" ]; then
    echo "This will perform a TARGETED cleanup."
    echo "The following directories and files will be DELETED:"
    echo "  - ./ddp_debug_checkpoints/ (and its contents in external storage)"
    echo "  - ./checkpoints/ (and its contents in external storage, for sweeps)"
    echo "  - ./training_logs/"
    echo "  - ./wandb/ (local offline logs)"
    echo "  - All *.log files in the root directory"
    echo "  - All experiment_summary.* and hybrid_experiment_summary.* files"
    echo "  - All temporary callback configs"
    echo ""
    echo "The following will be PRESERVED:"
    echo "  - ${EXTERNAL_STORAGE_PATH}/hybrid_checkpoints/"
    echo ""

    if confirm "Are you sure you want to proceed?"; then
        echo "--- Cleaning up experiment and debug artifacts... ---"
        
        # External Storage Directories
        rm -rf "${EXTERNAL_STORAGE_PATH}/ddp_debug_checkpoints"
        rm -rf "${EXTERNAL_STORAGE_PATH}/checkpoints"
        
        # Local Project Directories
        rm -rf ./training_logs/
        rm -rf ./wandb/
        
        # Local Files
        rm -f ./*.log
        rm -f ./experiment_summary.*
        rm -f ./hybrid_experiment_summary.*
        rm -f ./configs/callbacks/temp_*.yaml
        
        echo "--- Experiment cleanup complete. ---"
    else
        echo "Cleanup cancelled."
    fi

elif [ "$1" == "all" ]; then
    echo "This will perform a FULL cleanup."
    echo "WARNING: This will delete ALL generated artifacts, including ALL checkpoints"
    echo "from ALL runs (debug, experiments, and formal diagnostics)."
    echo ""
    
    if confirm "ARE YOU ABSOLUTELY SURE you want to delete everything?"; then
        echo "--- Performing full cleanup... ---"
        
        # External Storage Directories (All of them)
        rm -rf "${EXTERNAL_STORAGE_PATH}/ddp_debug_checkpoints"
        rm -rf "${EXTERNAL_STORAGE_PATH}/checkpoints"
        rm -rf "${EXTERNAL_STORAGE_PATH}/hybrid_checkpoints"
        
        # Local Project Directories
        rm -rf ./training_logs/
        rm -rf ./wandb/
        
        # Local Files
        rm -f ./*.log
        rm -f ./experiment_summary.*
        rm -f ./hybrid_experiment_summary.*
        rm -f ./configs/callbacks/temp_*.yaml

        # Pycache files
        find . -type d -name "__pycache__" -exec rm -rf {} +

        echo "--- Full cleanup complete. ---"
    else
        echo "Cleanup cancelled."
    fi

else
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  experiments   - Cleans up artifacts from debug runs and experiment sweeps."
    echo "                (Keeps 'hybrid_checkpoints' for formal runs)."
    echo "  all           - Deletes ALL generated artifacts, including all checkpoints,"
    echo "                logs, and local W&B data."
    echo ""
fi