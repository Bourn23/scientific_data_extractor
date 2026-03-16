#!/bin/bash

# =============================================================================
# Batch Visualization Script for Provenance Dashboard
# =============================================================================
# This script automates running visualize_provenance_dashboard.py on all 
# subdirectories within a given parent directory.
#
# Usage:
#   ./batch_visualize_provenance.sh <path_to_parent_folder>
#
# Example:
#   ./batch_visualize_provenance.sh "./output/downselectedpapers_jiyoung"
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
VISUALIZE_SCRIPT="$SCRIPT_DIR/03.visualize_provenance.py"

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
echo "Batch Visualizing Directory: $PARENT_FOLDER"
echo "Visualization Script: $VISUALIZE_SCRIPT"
echo "====================================================="

# Iterate through all subdirectories
for SUBDIR in "$PARENT_FOLDER"/*/; do
    # Skip if it's not a directory
    [ -d "${SUBDIR}" ] || continue
    
    # Remove trailing slash for cleaner output
    SUBDIR="${SUBDIR%/}"
    echo ""
    echo "Processing folder: $SUBDIR"
    
    # Find the provenance JSON file (should match *_provenance.json)
    PROV_JSON=$(find "$SUBDIR" -maxdepth 1 -name "*_provenance.json" -type f | head -n 1)
    
    if [ -z "$PROV_JSON" ]; then
        echo "[WARNING] No *_provenance.json file found in: $SUBDIR. Skipping."
        continue
    fi
    
    OUTPUT_HTML="$SUBDIR/provenance_dashboard.html"
    echo "Found provenance file: $PROV_JSON"
    echo "Output dashboard: $OUTPUT_HTML"
    
    # Run the visualization script
    if python "$VISUALIZE_SCRIPT" --input "$PROV_JSON" --out "$OUTPUT_HTML"; then
        echo "[SUCCESS] Finished generating dashboard for $(basename "$SUBDIR")"
    else
        echo "[ERROR] Failed generating dashboard for $(basename "$SUBDIR")"
    fi
done

echo ""
echo "====================================================="
echo "Batch visualization completed."
echo "====================================================="
