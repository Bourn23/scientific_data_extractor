#!/bin/bash

# =============================================================================
# Batch Processing Script for Extraction v8
# =============================================================================
# This script automates running the basic_extraction_md_v8.py script on all 
# subdirectories within a given parent directory.
#
# Usage:
#   ./batch_process_v8.sh <path_to_parent_folder>
#
# Example:
#   ./batch_process_v8.sh "./fetched_papers/obelix_md"
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

# Define Script Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
EXTRACTION_SCRIPT="$SCRIPT_DIR/01.extract_measurement.py"

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
echo "Extraction Script: $EXTRACTION_SCRIPT"
echo "====================================================="

# Iterate through all items in the parent directory
for SUBDIR in "$PARENT_FOLDER"/*/; do
    # Skip if it's not a directory
    [ -d "${SUBDIR}" ] || continue
    
    # Remove trailing slash for cleaner output
    SUBDIR="${SUBDIR%/}"
    echo ""
    echo "Processing folder: $SUBDIR"
    
    # Check if there is exactly one md file or just find the first one
    MD_FILE=$(find "$SUBDIR" -maxdepth 1 -name "*.md" -type f | head -n 1)
    
    if [ -z "$MD_FILE" ]; then
        echo "[WARNING] No .md file found in: $SUBDIR. Skipping."
        continue
    fi
    
    echo "Found MD file: $MD_FILE"
    
    # Run the extraction script
    export PYTHONUNBUFFERED=1
    if python -u "$EXTRACTION_SCRIPT" "$MD_FILE" --asset_dir "$SUBDIR"; then
        echo "[SUCCESS] Finished processing $MD_FILE"
    else
        echo "[ERROR] Failed processing $MD_FILE"
    fi
done

echo ""
echo "====================================================="
echo "Batch processing completed."
echo "====================================================="
