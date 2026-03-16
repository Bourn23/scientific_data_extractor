#!/bin/bash

# =============================================================================
# Batch Processing Script for Provenance Tracer
# =============================================================================
# This script automates running the t0_provenance_tracer.py script on all 
# subdirectories within a given parent directory.
#
# Usage:
#   ./batch_process_provenance.sh <path_to_parent_folder>
#
# Example:
#   ./batch_process_provenance.sh "./output/downselectedpapers_jiyoung"
# =============================================================================

set -e

# Initialize conda/mamba
CONDA_BASE="/Users/bourn23/miniforge3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
if [ -f "$CONDA_BASE/etc/profile.d/mamba.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/mamba.sh"
fi

# Mamba environment name
MAMBA_ENV="pokeagent"
mamba activate "$MAMBA_ENV"

# Helper function to run Python commands in the mamba environment
run_python() {
    export PYTHONUNBUFFERED=1
    python -u "$@"
}

# Define Script Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROVENANCE_SCRIPT="$SCRIPT_DIR/02.extract_provenance.py"

PARENT_FOLDER="$1"

if [ -z "$PARENT_FOLDER" ]; then
    echo "Usage: $0 <path_to_parent_folder>"
    exit 1
fi

if [ ! -d "$PARENT_FOLDER" ]; then
    echo "Error: Directory $PARENT_FOLDER does not exist."
    exit 1
fi

echo "====================================================="
echo "Batch Processing Directory: $PARENT_FOLDER"
echo "Provenance Script: $PROVENANCE_SCRIPT"
echo "====================================================="

# Ensure absolute path for the v8 directory so the tracer finds it
ABS_PARENT_FOLDER=$(cd "$PARENT_FOLDER" && pwd)

# We can reuse the extraction logic where we run the tracer on the entire folder.
# We don't need a loop because the script already discovers papers in a given directory!!
# Actually, looking at discover_papers() in the python script, we just need to pass the parent directory as --v8-dir.
# Wait, the python script uses PAPERS_DIR internally which is hardcoded. Let's pass the specific folder as a sample,
# looping through to make sure we hit every paper explicitly, or modify the Python script slightly.
# Since the python script takes `--sample "folder_name"`, we can loop through the parent folder and pass it.

for SUBDIR in "$PARENT_FOLDER"/*/; do
    # Skip if it's not a directory
    [ -d "${SUBDIR}" ] || continue
    
    # Remove trailing slash for cleaner output
    SUBDIR="${SUBDIR%/}"
    FOLDER_NAME=$(basename "$SUBDIR")
    echo ""
    echo "Processing paper: $FOLDER_NAME"
    
    # Run the provenance script on this specific sample
    export PYTHONUNBUFFERED=1
    
    # We pass --v8-dir explicitly to the parent folder so it knows where to look.
    # The tracer script also uses PAPERS_DIR which we will need to ensure it finds.
    # Actually, the tracer script hardcodes PAPERS_DIR to /Users/bourn23/Downloads/general/PageIndex/output/downselectedpapers_jiyoung
    # Let me run it and see. If it fails, we will need to modify the python script to accept PAPERS_DIR as an argument.
    
    if run_python "$PROVENANCE_SCRIPT" --sample "$FOLDER_NAME" --papers-dir "$ABS_PARENT_FOLDER" --v8-dir "$ABS_PARENT_FOLDER"; then
        echo "[SUCCESS] Finished tracing provenance for $FOLDER_NAME"
    else
        echo "[ERROR] Failed tracing provenance for $FOLDER_NAME"
    fi
done

echo ""
echo "====================================================="
echo "Batch processing completed."
echo "====================================================="
