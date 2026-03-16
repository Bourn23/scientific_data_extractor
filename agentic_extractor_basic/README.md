# Agentic Extractor (Basic)

## Overview

This is the **agentic-only** implementation of the scientific data extraction framework. It uses a multi-agent orchestration system (Google Agentic Development Kit) where specialized LLM agents work together to extract and validate ionic conductivity measurements from scientific literature.

### How It Works

The pipeline employs four specialized agents that coordinate to extract data:

1. **Context Librarian**
   - Reads the introduction and methods sections
   - Builds a comprehensive abbreviation map
   - Maps sample labels (e.g., "LATP03") to canonical formulas (e.g., "Li1.3Al0.3Ti1.7(PO4)3")

2. **Data Extractor**
   - Extracts raw conductivity data from figures and tables
   - Handles Arrhenius plots, conductivity vs. temperature curves, and markdown tables
   - Returns measurements with raw values and extracted metadata

3. **Materials Chemist**
   - Validates chemical formulas and stoichiometry
   - Resolves material names using the abbreviation map
   - Normalizes units and checks physical bounds
   - Flags suspicious or out-of-range values

4. **Lead Researcher** (Root Agent)
   - Orchestrates the entire pipeline
   - Coordinates between other agents
   - Applies deduplication and source preference rules
   - Compiles the final JSON report

Unlike the `agentic_extractor_with_harness`, this version **does not include a verifier loop** — it focuses on LLM-driven extraction and validation without iterative refinement.

## Prerequisites

Before running the extractor, ensure you have:

1. **Markdown file** with embedded images (converted via [markitdown](https://github.com/microsoft/markitdown))
2. **Gemini API key** configured in `.env` file
3. **Python environment** set up with dependencies (see main README for setup instructions)

## Usage

### Basic Usage

Run the extraction pipeline:

```bash
python agent.py <path_to_paper.md>
```

The script will:
- Parse the markdown document structure
- Extract abbreviations from the introduction/methods
- Extract measurements from figures and tables
- Validate and normalize all data
- Output a JSON report to the console

### Example

```bash
# Activate your virtual environment
source env/bin/activate

# Convert PDF to markdown (if needed)
python -m markitdown paper.pdf > paper.md

# Run the extraction pipeline
python agent.py ./paper.md
```

## Output Format

The agent returns a JSON report with this structure:

```json
{
  "doc_title": "Paper title",
  "total_measurements": 42,
  "measurements": [
    {
      "raw_composition": "LATP03",
      "canonical_formula": "Li1.3Al0.3Ti1.7(PO4)3",
      "material_class": "Ceramic",
      "normalized_conductivity": 1.2e-4,
      "normalized_temperature_c": 25.0,
      "source_type": "figure",
      "confidence": "high",
      "warnings": []
    }
  ]
}
```

## When to Use This Version

Use `agentic_extractor_basic` when:

- You need **fast extraction** without verification overhead
- You don't need **hallucination detection**
- You're **not concerned** about missing data recovery
- You want a **pure LLM-driven** approach

For **production use** or when you need high-quality results with verification, use `agentic_extractor_with_harness` instead.

## Key Differences from `agentic_extractor_with_harness`

| Feature | Basic | With Harness |
|---------|-------|--------------|
| **Approach** | Pure LLM agents | Deterministic + Verifier |
| **Verification** | Validation only | Iterative refinement |
| **Hallucination Detection** | Limited | Advanced (PDF-based) |
| **Speed** | Fast | Medium |
| **Quality** | Good | High (recommended) |
| **PDF Required** | No | Yes (recommended) |
