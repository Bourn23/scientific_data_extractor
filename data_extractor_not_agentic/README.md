# Non-Agentic Data Extractor

The **data_extractor_not_agentic** folder contains a sequential pipeline for extracting ionic conductivity data from scientific literature without using LLM agents. This is the baseline approach that provides a simple, deterministic extraction workflow.

## Pipeline Overview

The pipeline has two main workflows:

### Workflow 1: Markdown-based Extraction (Recommended)
A multi-stage sequential pipeline that processes markdown-converted PDFs:

1. **01.extract_measurement.py** — Extract ionic conductivity measurements
2. **02.extract_per_system_provenance.py** — Extract fabrication process provenance
3. **03.extract_material_features.py** — Extract material identity features
4. **04.t1_tier_pipeline.py** — Perform tier-based analysis

### Workflow 2: Direct PDF Extraction (Baseline)
A simplified single-step extraction that works directly on PDF files:

- **extract_from_pdf.py** — Extract measurements directly from PDF (no markdown preprocessing)

---

## Detailed Step-by-Step Guide

### Prerequisites

Ensure your mamba environment is activated:
```bash
mamba activate pokeagent
```

And you have set up your `.env` file with your Gemini API key (see main repository README).

### Step 1: Extract Measurement Data

**File:** `01.extract_measurement.py`

Extracts ionic conductivity measurements from your markdown file, processing both text and images.

**Usage:**
```bash
python 01.extract_measurement.py /path/to/your/document.md
python 01.extract_measurement.py /path/to/your/document.md --asset_dir /path/to/images/
python 01.extract_measurement.py /path/to/your/document.md --model gemini-3-flash-preview
```

**Output:** `robust_results_v8.json` (in the same directory as your markdown file)

Contains:
- Extracted ionic conductivity measurements
- Material definitions
- Paper context and metadata
- Cost summary and extraction statistics

---

### Step 2: Extract Per-System Provenance

**File:** `02.extract_per_system_provenance.py`

Extracts the fabrication process provenance (3-phase narrative) for each material system identified in the measurements.

**Usage:**
```bash
# Process a specific paper
python 02.extract_per_system_provenance.py --paper "Your Paper Name"

# Process all papers in the papers directory
python 02.extract_per_system_provenance.py --all

# Force reprocessing (overwrite existing results)
python 02.extract_per_system_provenance.py --paper "Your Paper Name" --force

# Use a different model
python 02.extract_per_system_provenance.py --paper "Your Paper Name" --model gemini-2.5-flash
```

**Input Requirements:**
- Markdown file in a paper folder
- `robust_results_v8.json` (output from Step 1)

**Output:** `process_paragraphs_v2_llm_grouping.json` (in the paper folder)

Contains:
- Material system groupings and assignments
- 3-phase fabrication process narratives per system

---

### Step 3: Extract Material Features

**File:** `03.extract_material_features.py`

Extracts material identity features (polymer class, ceramic type, salt type, filler loading, morphology) from the provenance output.

**Usage:**
```bash
# Process a specific paper
python 03.extract_material_features.py --paper "Your Paper Name"

# Process all papers
python 03.extract_material_features.py --all

# Force reprocessing with a different model
python 03.extract_material_features.py --all --force --model gemini-2.5-flash
```

**Input Requirements:**
- `robust_results_v8.json` (from Step 1)
- `process_paragraphs_v2_llm_grouping.json` (from Step 2)

**Outputs:**
- Per-paper: `T0_structured_features.csv` (inside each paper folder)
- Consolidated: `T0_structured_features.csv` in the output directory (all papers merged)

Contains:
- Polymer class (PEO, PVDF, PAN, etc.)
- Ceramic type (LLZO, LATP, etc.)
- Salt type (LiTFSI, LiClO4, etc.)
- Filler loading (wt%)
- Filler morphology

---

### Step 4: Tier-Based Analysis

**File:** `04.t1_tier_pipeline.py`

Performs data-driven tier classification and synergy score analysis for polymer-ceramic composite electrolytes.

**Usage:**
```bash
# Basic analysis
python 04.t1_tier_pipeline.py --input /path/to/T0_structured_features.csv --output-dir ./t1_output/

# With custom threshold
python 04.t1_tier_pipeline.py --input /path/to/T0_structured_features.csv --output-dir ./t1_output/ --threshold 0.3
```

**Inputs:**
- `T0_structured_features.csv` (from Step 3)

**Outputs:**
- `T1_tiered_dataset.csv` — Tier classifications and baseline sources
- `T1_tier_summary.json` — Summary statistics
- Visualization plots

Contains:
- 3-tier baseline hierarchy (same-paper > dataset-wide > literature fallback)
- wt% to vol% conversions with material densities
- Synergy scores and tier assignments

---

## Alternative: Direct PDF Extraction

**File:** `extract_from_pdf.py`

Use this simplified baseline if you want to extract measurements directly from a PDF file without converting to markdown first.

**Advantages:**
- Single step extraction
- No markdown conversion required
- Good for quick benchmarking

**Disadvantages:**
- No provenance tracking
- No detailed material feature extraction
- Less detailed output

**Usage:**
```bash
# Single PDF
python extract_from_pdf.py paper.pdf

# Batch process a directory
python extract_from_pdf.py papers_dir/ --batch

# With specific model and thinking level (Gemini 3 only)
python extract_from_pdf.py paper.pdf --model gemini-3-flash-preview --thinking-level high
```

**Supported Models:**
- Gemini 2.5: `gemini-2.5-flash` (default), `gemini-2.5-flash-lite`, `gemini-2.5-pro`
- Gemini 3: `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`

**Thinking Levels (Gemini 3 only):** `minimal`, `low`, `medium`, `high`

**Output:** `measurements.json` in the same directory as the PDF

---

## Complete Workflow Example

```bash
# 1. Extract measurements from markdown
python 01.extract_measurement.py example_data/my_paper.md

# 2. Extract per-system provenance
python 02.extract_per_system_provenance.py --paper "My Paper Name"

# 3. Extract material features
python 03.extract_material_features.py --paper "My Paper Name"

# 4. Run tier analysis
python 04.t1_tier_pipeline.py --input example_data/T0_structured_features.csv --output-dir ./results/
```

Or, for quick baseline extraction from PDF:
```bash
python extract_from_pdf.py example_data/my_paper.pdf
```

---

## Key Characteristics

- **Deterministic:** Produces consistent results without agent-based iteration
- **No verifier loop:** Does not automatically detect and recover from hallucinations
- **Simple:** Good for understanding the fundamental extraction pipeline
- **Useful for comparison:** Serves as a baseline for evaluating agentic approaches

For more advanced capabilities including hallucination detection and iterative refinement, see the agentic implementations in the parent repository.
