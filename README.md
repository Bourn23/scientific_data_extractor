# Scientific Data Extraction Framework

## Overview

This framework extracts **ionic conductivity** data from scientific literature in PDF format. It converts markdown-processed PDFs (via [markitdown](https://github.com/microsoft/markitdown)) into structured ionic conductivity measurements with full provenance tracking.

## Project Structure

The framework includes three implementations, each building on the previous approach:

### 1. **data_extractor_not_agentic/** (Baseline)
The simplest implementation—no LLM agents involved. Provides basic extraction with evaluation and retry logic but does not attempt to recover from failed extraction attempts. Serves as a reference implementation for understanding the data structures and pipeline flow.

### 2. **agentic_extractor_basic/** (Agentic, without verifier)
Adds LLM-driven extraction using agents. Includes evaluation and retry mechanisms, but lacks a verifier loop. Useful for faster extraction when hallucination detection and missing data recovery are not critical.

### 3. **agentic_extractor_with_harness/** (Latest - Agentic with Verifier Loop)
The most advanced implementation featuring a **verifier agent** that:
- Reads extraction logs and makes decisions to improve results
- Compares extracted data against the original PDF markdown
- Detects hallucinations and missing data points
- Iteratively refines extraction through feedback loops

The verifier loop is crucial for high-quality extractions and is recommended for production use.

## Getting Started

### Environment Setup

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Prerequisites

Before running the pipeline, convert your PDF to markdown format with embedded images:

1. **Convert PDF to Markdown** using [markitdown](https://github.com/microsoft/markitdown):
   ```bash
   python -m markitdown <pdf_file> > document.md
   ```

2. **Image Format**: Ensure images extracted from the PDF follow this naming convention:
   ```
   _page_{page_number}_figure_{figure_number}.jpeg
   ```

   Example: `_page_3_figure_6.jpeg` (image from page 3, 6th figure)

   The extraction pipeline uses this naming format to correlate images with their source pages and figures for accurate data extraction and verification.

3. **Set up Gemini API Key**: The extraction pipelines use Google Gemini API for LLM-powered data extraction and analysis.

   - Get your API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
   - Copy `.env.template` to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace `your_api_key_here` with your actual Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - The `.env` file is ignored by git (see `.gitignore`) to keep your API key secure

## Extracting Ionic Conductivity Data

Choose one of the three extraction methods below. All produce compatible output formats for downstream analysis.

### Method 1: Non-Agentic (Baseline)

The simplest approach using deterministic extraction logic.

```bash
cd data_extractor_not_agentic
python 01.extract_measurement.py <path_to_paper.md>
```

**Output**: JSON file with extracted measurements

**Best for**: Understanding the data structures, fast baseline extraction, minimal API costs

---

### Method 2: Agentic Basic

LLM-driven extraction using multi-agent orchestration (fast, no verifier loop).

```bash
cd agentic_extractor_basic
python agent.py <path_to_paper.md>
```

**Output**: JSON report with validated measurements

**Best for**: Faster extraction when you don't need hallucination detection

---

### Method 3: Agentic with Verifier (Recommended for Production)

Two-phase pipeline: deterministic extraction + intelligent verifier agent for high-quality results.

```bash
cd agentic_extractor_with_harness
python agent.py <path_to_paper.md> [path_to_paper.pdf]
```

**Output**: `robust_results_v8.json` with extracted and verified measurements

**Best for**: Production use, high-quality results, hallucination detection, missing data recovery

---

## Downstream Analysis

Once you've extracted ionic conductivity data using any of the three methods, you can perform downstream analysis:

### Extract Process Parameters

Use the extracted measurements to derive process parameters and material features:

```bash
cd data_extractor_not_agentic

# Step 1: Extract system provenance
python 02.extract_per_system_provenance.py --input <extracted_data.json>

# Step 2: Extract material features
python 03.extract_material_features.py --input <extracted_data.json>

# Step 3: Run T1 tier analysis
python 04.t1_tier_pipeline.py --input <extracted_data.json> --output-dir <output_dir>
```

**Note**: The scripts in `data_extractor_not_agentic/` work with extracted data from **any** method (01, agentic_extractor_basic, or agentic_extractor_with_harness). Just use the JSON output from your chosen extraction method as input.

### Output Files

- Provenance data with system-level information
- Material features (compositions, properties, processing conditions)
- T1 tiered analysis with categorized measurements
