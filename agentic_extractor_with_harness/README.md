# Agentic Extractor with Verifier Loop

## Overview

This is the **production-ready** implementation of the scientific data extraction framework. It combines deterministic data extraction with an intelligent verifier agent to ensure high-quality, hallucination-resistant results.

### How It Works

The pipeline operates in **two phases**:

1. **Phase 1: Deterministic Extraction Pipeline**
   - Runs the core extraction logic from the non-agentic version
   - Systematically extracts ionic conductivity measurements from the markdown document
   - Applies normalization and validation rules
   - Saves results to `robust_results_v8.json`

2. **Phase 2: Verifier Agent**
   - Reads the extraction logs and results from Phase 1
   - Processes the original PDF document for verification
   - Detects hallucinations and missing data points
   - Iteratively refines extraction through feedback loops
   - Provides a detailed verification report

This two-phase approach gives you both speed (Phase 1) and quality assurance (Phase 2).

## Prerequisites

Before running the extractor, ensure you have:

1. **Markdown file** with embedded images (converted via [markitdown](https://github.com/microsoft/markitdown))
2. **PDF file** (highly recommended for verifier accuracy) — must be in the same directory as the markdown file or provided as an argument
3. **Gemini API key** configured in `.env` file
4. **Python environment** set up with dependencies (see main README for setup instructions)

## Usage

### Basic Usage

Run the extraction and verification pipeline:

```bash
python agent.py <path_to_paper.md>
```

The script will:
- Auto-detect the PDF file if it's in the same directory as the markdown file
- Run Phase 1 (extraction)
- Run Phase 2 (verification)
- Save results to `robust_results_v8.json`

### With Explicit PDF Path

```bash
python agent.py <path_to_paper.md> <path_to_paper.pdf>
```

## Output Files

After running the pipeline, you'll get:

- **`robust_results_v8.json`** — Main extraction results including:
  - Extracted measurements with raw and normalized values
  - Material definitions
  - Extraction statistics
  - API cost summary
  - Paper metadata

- **`extraction_log.jsonl`** — Detailed logs of all extraction operations (used by verifier)

- **Verifier Report** — Printed to console showing verification results and any recovered data

## Important Notes

- **PDF Required for Full Verification**: While the extraction phase works with just the markdown file, the verifier agent uses the PDF to verify results and detect hallucinations. For best results, always provide the PDF file.

- **Temperature Assumptions**: If a measurement is missing temperature data, the pipeline assumes room temperature (25°C) and flags this in the warnings field.

- **API Costs**: The `robust_results_v8.json` includes a cost summary showing input/output tokens and estimated USD cost.

## Example Workflow

```bash
# Activate your virtual environment
source env/bin/activate

# Convert PDF to markdown (if needed)
python -m markitdown paper.pdf > paper.md

# Run the extraction and verification pipeline
python agent.py ./paper.md

# Check the results
cat robust_results_v8.json
```
