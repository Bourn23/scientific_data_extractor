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

All implementations expect markdown-converted PDF content as input. Use [markitdown](https://github.com/microsoft/markitdown) to convert your PDF:

```bash
python -m markitdown <pdf_file> > document.md
```

Then pass the markdown to your chosen extractor implementation.
