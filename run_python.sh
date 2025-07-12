#!/bin/bash

# This script executes a command, ensuring that any call to 'python'
# uses the specific python from the 'gfn' conda environment.

echo "Robust Wrapper v2 is executing..."

# Define the absolute path to the target python executable
PYTHON_EXE="/hpc2hdd/home/zhongal/miniconda3/envs/gfn/bin/python"

# Create a new array to hold the modified command
CMD_ARGS=()

# Loop through all arguments passed to this script
for arg in "$@"; do
    # If the argument is exactly "python", replace it with the full path
    if [[ "$arg" == "python" ]]; then
        CMD_ARGS+=("$PYTHON_EXE")
    else
        # Otherwise, keep the argument as is
        CMD_ARGS+=("$arg")
    fi
done

# Execute the reconstructed command, which now has the full path to python
exec "${CMD_ARGS[@]}"