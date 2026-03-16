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

4. **Pass to Extractor**: Use the markdown file and its associated images with your chosen implementation (data_extractor_not_agentic, agentic_extractor_basic, or agentic_extractor_with_harness).
