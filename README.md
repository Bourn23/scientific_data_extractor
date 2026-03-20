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

3. **Download spacy language model** (required for text processing):
   ```bash
   python -m spacy download en_core_web_sm
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


## Technical Details/Architecture
# Extraction Pipeline — Technical Overview

This document describes the automated extraction pipeline used to pull ionic conductivity measurements from academic papers on polymer-ceramic composite solid electrolytes. It is written for collaborators who are presenting this work and need a solid understanding of what the system does and how, without needing to read the code directly.

---

## What the Pipeline Does

The pipeline reads academic papers (converted to Markdown with embedded images) and extracts structured ionic conductivity measurements — composition, conductivity value, temperature, processing method, and provenance — from three modalities simultaneously: **text**, **figures**, and **tables**. All results are merged, deduplicated, normalized, and saved to a single JSON file per paper.

- Input: a Markdown file (from PDF conversion) + a folder of figure images from the same paper.
- Output: `robust_results_v8.json` — a structured list of measurements, each with full provenance and normalized values in S/cm and °C.
- All modalities are processed in parallel using async calls, then merged in a single post-processing chain.

---

## Step 0 — Paper-Level Context Extraction (runs once before everything else)

Before any figures or text sections are processed, the pipeline runs three LLM calls on the paper's text to build a shared context object that gets injected into all downstream extraction calls.

- **Abbreviation map**: the model reads the abstract, introduction, and experimental sections and builds a mapping from paper-specific shorthand to full material descriptions — e.g., `"CPE-10"` → `"PEO-LiTFSI/Li6.4Al0.2La3Zr2O12 (10 wt%) [Composite]"`. This is the highest-reliability resolution path used later.
- **Processing map**: the model reads the experimental/methods sections and maps each sample label to its synthesis method — e.g., `"LATP"` → `"solid state reaction, sintered at 900°C in air"`.
- **Paper context summary**: a five-field structured summary is extracted — experimental procedure narrative, sample nomenclature key, material systems overview, measurement setup, and champion/baseline samples. The baseline sample is the raw/initial reported value that the authors can compare their results to. This is injected as a preamble into every text section and table extraction call so the model understands cross-section context (e.g., processing conditions described in methods can be linked to data reported in results).

---

## Modality 1 — Text Extraction

Each section of the paper is sent to the LLM independently. Before sending, each paragraph is numbered (`[P1]`, `[P2]`, ...) so the model can cite exactly which paragraph a value came from.

- The prompt enforces a strict distinction between **measured** data (`source="text"`) and **cited** data (`source="cited_text"`). Any conductivity value that appears alongside a reference number like `[12]` is tagged as cited and later flagged for ML exclusion if it comes from an introduction or review section.
- The model is asked to extract each time point of aging/stability data as a **separate measurement** and to split conductivity ranges (e.g., "10⁻² to 10⁻⁴ S/cm") into two measurements tagged `range_upper_bound` / `range_lower_bound`.
- After extraction, the paragraph indices the model reports are used to include the actual paragraph text into `source_paragraph_text` field, giving reviewers a direct quote for every measurement.

---

## Modality 2 — Figure Extraction (two-stage vision pipeline)

Figure processing is a two-LLM-call pipeline per image, not one.

### Stage 1 — Subplot Detection

The full raw figure image is sent to Gemini 3 Flash (vision model) with a prompt asking it to locate all subplots and classify each one. The model returns normalized bounding boxes (0–1000 coordinate space), axis metadata (type, unit, scale, value range), and legend labels for each panel.

- Subplots are **filtered before extraction**: frequency-axis plots (impedance spectroscopy), Nyquist plots, XRD diagrams, and structural schematics are discarded. Only plots where the Y-axis is conductivity and the X-axis is temperature or stoichiometry/composition proceed.
- The detection result provides `axis_hints` — a structured dict of axis titles, units, quantity types (e.g., `temperature_inverse`, `stoichiometry`), and value ranges — that gets passed directly into Stage 2's prompt.

### Stage 2 — Data Extraction from Cropped Subplot

Each valid subplot is physically cropped from the original image using the detected bounding box (with 80px padding). The crop is then sent to a second LLM call.

- The cropped image is sent **alongside a grid of sub-patches** (2×2 for linear-scale plots, 3×3 for log-scale Arrhenius plots) in the same API call. The full crop provides overview context; the patches provide higher effective resolution for reading tick marks and dense data clusters.
- The axis hints from Stage 1 are injected as text into the Stage 2 prompt: `X-AXIS: 1000/T (K⁻¹) [Range: 2.5 to 4.0]`, `LEFT Y-AXIS: log(σ) [Range: -7 to -2]`. This grounds the model in the correct scale and prevents misinterpretation of log-scale axes.
- **Dual Y-Axis Handling**: The pipeline explicitly maps data series to either a primary (`left`) or secondary (`right`) Y-axis. This allows for simultaneous extraction of different metrics (like conductivity and activation energy) from a single cross-plotted figure.
- **Contextual Legend Interpretation**: The prompt engineering explicitly instructs the LLM to interpret legend labels differently based on the X-axis type—e.g., if the plot is frequency (Hz), labels like '223K' are treated as experimental temperatures; if it's an Arrhenius plot (1000/T), labels are treated as material variants.
- The model outputs parallel `x_values[]` / `y_values[]` lists per named series (using exact legend text, not generic "Series 1" labels), plus `annotated_temperature` if temperature appears as an annotation inside the plot rather than on an axis. Extracted values are validated against the axis ranges (with 15% margin) and out-of-bounds points are removed before the result is accepted.

---

## Modality 3 — Table Extraction

Markdown tables are detected by regex and **removed from the text** before text sections are sent to the LLM (replaced with a placeholder), so the same data is never extracted twice.

- The table caption and raw Markdown content are sent together. The prompt explicitly handles **multi-level column headers** (e.g., "BULK" / "Grain Boundary" spanning sub-columns for σ₀, Eₐ, σRT) and instructs the model to prefer σRT (room-temperature conductivity) over σ₀ (the pre-exponential factor, which is orders of magnitude larger and not a conductivity).
- Bulk and grain boundary values from the same row are extracted as separate measurements. The `paper_context` preamble is prepended so the model can fill in processing method even when it is described only in the experimental section, not repeated in the table.
- Temperature is set to `0.0` for table calls (fully deterministic), compared to `0.7–1.0` for text and figure calls where some interpretive flexibility is needed.

---

## Post-Processing — Merging, Resolving, and Normalizing

After all parallel tasks complete, a sequential post-processing chain runs on the merged measurement list.

### Name Resolution (`canonicalize_materials`)
Raw composition strings from figures and text (e.g., `"x=0.3"`, `"LATP03"`, `"Sample A"`) are resolved to full canonical formulas in three stages:
1. **Abbreviation map lookup** — the paper-specific map built in Step 0 (highest reliability).
2. **Deterministic NASICON resolver** — rule-based decoder for LATP/LCTP/LFTP acronyms (e.g., `"LATP03"` → `"Li1.3Al0.3Ti1.7(PO4)3"`).
3. **LLM call** — for anything remaining, using document title + abstract + material definitions extracted from text.

### Deduplication
Text measurements are grouped by `(normalized_conductivity, normalized_temperature_c)`. Within each group, the best entry is kept using the priority: has canonical formula > higher confidence > longer formula > earlier in document.

### Unit Normalization
All conductivity values are converted to S/cm and all temperatures to °C. The normalizer handles: log(σ), ln(σ), σ·T (Arrhenius pre-exponential form), mS/cm, μS/cm, S/m, and inverse-temperature axes (1000/T → °C conversion).

### ML Exclusion Flagging
Cited measurements extracted from introduction or conclusion sections are marked `exclude_from_ml = True`. These are literature-comparison values lacking direct processing metadata and should not be used as training data for the downstream ML pipeline.

### Missing Temperature Assumption
Measurements with a valid conductivity but no temperature (common for composition-vs-conductivity plots without an annotated temperature) are assigned 25°C with a warning appended.

---

## Output Format

Each paper produces `robust_results_v8.json` containing:

- **`measurements[]`** — the full list of `MeasuredPoint` records. Key fields per record:

  | Field | Description |
  |---|---|
  | `raw_composition` | Name exactly as extracted from the paper |
  | `canonical_formula` | Resolved full formula (e.g., `"PEO-LiTFSI/Li6.4Al0.2La3Zr2O12 (50 wt%)"`) |
  | `material_class` | `Polymer` / `Ceramic` / `Composite` |
  | `normalized_conductivity` | Value in S/cm |
  | `normalized_temperature_c` | Temperature in °C |
  | `processing_method` | Synthesis route (e.g., `"solution casting, dried at 60°C"`) |
  | `source` | `text` / `figure` / `table` / `cited_text` / `cited_figure` / `cited_table` |
  | `source_section` | Section of the paper (e.g., `"Results and Discussion"`) |
  | `source_paragraph_text` | Exact paragraph(s) the value was extracted from |
  | `source_figure_id` | Figure number (e.g., `"Fig 4b"`) |
  | `exclude_from_ml` | `true` if cited from intro/review, should not enter ML training |
  | `confidence` | `high` / `medium` / `low` |
  | `warnings[]` | List of any issues flagged during extraction or normalization |

- **`paper_context`** — the five-field paper summary (procedure, nomenclature, setup, etc.) for downstream use.
- **`cost_summary`** — total token counts and estimated USD cost per model.
- **`extraction_stats`** — count of measurements extracted, images skipped, and tasks that failed.

