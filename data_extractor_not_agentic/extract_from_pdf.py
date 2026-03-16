"""
Simplified PDF-based ionic conductivity extractor.
This can be used as a benchmark for the agentic extractor.
Replaces the multi-stage markdown pipeline with a single LLM call on the raw PDF.

Usage:
    python extract_from_pdf.py paper.pdf
    python extract_from_pdf.py paper.pdf --model gemini-3-flash-preview
    python extract_from_pdf.py paper.pdf --model gemini-3.1-pro-preview --thinking-level low
    python extract_from_pdf.py papers_dir/ --batch --model gemini-3.1-flash-lite-preview

Supported models:
  Gemini 2.5:  gemini-2.5-flash (default), gemini-2.5-flash-lite, gemini-2.5-pro
  Gemini 3:    gemini-3-flash-preview, gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview

Thinking levels (Gemini 3 only): minimal, low, medium, high
  Defaults: gemini-3.1-flash-lite-preview → minimal
            gemini-3-flash-preview / gemini-3.1-pro-preview → high
"""

import re
import json
import math
import time
import argparse
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Model registry ───────────────────────────────────────────────────────────

# Gemini 3 models require v1alpha API, inline PDF data, thinking_config, and temp=1.0 (default)
GEMINI3_MODELS = {
    "gemini-3.1-flash-lite-preview": {"default_thinking": "minimal", "input_usd": 0.25,  "output_usd": 1.50},
    "gemini-3-flash-preview":        {"default_thinking": "high",    "input_usd": 0.50,  "output_usd": 3.00},
    "gemini-3.1-pro-preview":        {"default_thinking": "high",    "input_usd": 2.00,  "output_usd": 12.00},
}

GEMINI2_MODELS = {
    "gemini-2.5-flash-lite":   {"input_usd": 0.10, "output_usd": 0.40},
    "gemini-2.5-flash":        {"input_usd": 0.30, "output_usd": 2.50},
    "gemini-2.5-flash-latest": {"input_usd": 0.30, "output_usd": 2.50},
    "gemini-2.5-pro":          {"input_usd": 1.25, "output_usd": 10.00},
}

def is_gemini3(model_name: str) -> bool:
    return model_name in GEMINI3_MODELS

def estimate_cost(model_name: str, in_tok: int, out_tok: int) -> float:
    pricing = GEMINI3_MODELS.get(model_name) or GEMINI2_MODELS.get(model_name, {"input_usd": 0.30, "output_usd": 2.50})
    return (in_tok / 1_000_000 * pricing["input_usd"]) + (out_tok / 1_000_000 * pricing["output_usd"])

# ── Schema ────────────────────────────────────────────────────────────────────

class MeasuredPoint(BaseModel):
    raw_composition: str = Field(description="Full material name as written (e.g. 'PEO-LiTFSI/LLZO (50 wt%)')")
    canonical_formula: Optional[str] = Field(default=None, description="Resolved chemical formula")
    material_class: Optional[str] = Field(default=None, description="Polymer, Ceramic, or Composite")
    material_definition: Optional[str] = Field(default=None, description="1-2 sentence description of the material")
    raw_conductivity: str = Field(description="Numeric conductivity value (e.g. '1.2e-4')")
    raw_conductivity_unit: str = Field(description="Unit (e.g. 'S/cm', 'mS/cm')")
    normalized_conductivity: Optional[float] = Field(default=None)
    raw_temperature: Optional[str] = Field(default=None, description="Temperature value (e.g. '25', '298')")
    raw_temperature_unit: Optional[str] = Field(default=None, description="Unit (e.g. 'Celsius', 'K', '1000/T (K-1)')")
    normalized_temperature_c: Optional[float] = Field(default=None)
    source: str = Field(description="text | cited_text | figure | cited_figure | table | cited_table")
    source_detail: Optional[str] = Field(default=None, description="Section name, figure ID, or table ID")
    processing_method: Optional[str] = Field(default=None, description="Synthesis/fabrication method")
    aging_time: Optional[str] = Field(default=None)
    measurement_condition: Optional[str] = Field(default=None)
    confidence: str = Field(default="medium", description="high, medium, or low")
    warnings: List[str] = Field(default_factory=list)
    exclude_from_ml: bool = Field(default=False)

class ExtractionResult(BaseModel):
    measurements: List[MeasuredPoint]

# ── Unit normalization ────────────────────────────────────────────────────────

def normalize_units(m: MeasuredPoint) -> None:
    """Normalize conductivity to S/cm and temperature to Celsius in-place."""
    if m.normalized_conductivity is not None and m.normalized_temperature_c is not None:
        return

    temp_c = temp_k = None
    try:
        raw_t_str = (m.raw_temperature or "").strip().lower()
        if raw_t_str not in ("", "null", "none", "n/a", "unknown", "not specified"):
            if raw_t_str in ("room temperature", "rt", "room temp", "ambient"):
                raw_t, unit = 25.0, "c"
            else:
                num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_t_str)
                raw_t = float(num.group()) if num else None
                unit = (m.raw_temperature_unit or "").lower().strip()

            if raw_t is not None:
                if ("1000" in unit or "10^3" in unit) and "t" in unit:
                    temp_k = 1000.0 / raw_t if raw_t > 0 else None
                elif ("k-1" in unit or "1/k" in unit) and 0.2 < raw_t < 10.0:
                    temp_k = 1000.0 / raw_t
                elif "k" in unit and "c" not in unit:
                    temp_k = raw_t
                elif "c" in unit:
                    temp_c = raw_t
                elif raw_t > 200:
                    temp_k = raw_t
                else:
                    temp_c = raw_t

                if temp_k is not None and temp_c is None:
                    temp_c = temp_k - 273.15
                if temp_c is not None and temp_k is None:
                    temp_k = temp_c + 273.15
    except Exception:
        pass

    m.normalized_temperature_c = round(temp_c, 2) if temp_c is not None else None

    try:
        raw_c_str = (m.raw_conductivity or "").strip().lower()
        if raw_c_str in ("", "null", "none", "n/a"):
            return
        num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_c_str)
        raw_c = float(num.group()) if num else None
        if raw_c is None:
            return
        cu = (m.raw_conductivity_unit or "").lower().strip()
        if any(x in cu for x in ("ev", "kj", "joule", "mol")):
            return  # activation energy, not conductivity
        if "log" in cu:
            m.normalized_conductivity = (10 ** raw_c) / temp_k if ("t" in cu and temp_k) else 10 ** raw_c
        elif "ln" in cu:
            m.normalized_conductivity = math.exp(raw_c) / temp_k if ("t" in cu and temp_k) else math.exp(raw_c)
        else:
            mul = 1e-3 if "ms" in cu else (1e-6 if ("μs" in cu or "us" in cu) else (1e-9 if "ns" in cu else 1.0))
            if "m" in cu and "cm" not in cu and ("m-1" in cu or "/m" in cu):
                mul *= 0.01
            m.normalized_conductivity = raw_c * mul
    except Exception:
        m.normalized_conductivity = None

# ── Prompt ────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an expert materials scientist. Extract ALL ionic conductivity measurements from this PDF of a solid-state electrolyte paper.

Return a JSON object with a single key "measurements", containing a list of measurement objects.

For EACH measurement found (in text, tables, AND figures), extract:

1. **raw_composition**: Full material name as written (e.g. "PEO-LiTFSI/LLZO (50 wt%)", "Li6.4La3Zr1.4Ta0.6O12").
   - For polymer composites: include polymer, salt, ceramic filler, and loading.
2. **canonical_formula**: Resolved chemical formula if determinable. Otherwise null.
3. **material_class**: "Polymer" (neat polymer electrolyte), "Ceramic" (neat ceramic), or "Composite" (polymer+ceramic with nonzero filler). Classify by what the material IS.
4. **material_definition**: 1-2 sentences describing the material (composition, structure, processing).
5. **raw_conductivity**: Numeric value only (e.g. "1.2e-4"). No ~ or < or > symbols.
6. **raw_conductivity_unit**: Unit exactly as presented (e.g. "S/cm", "mS/cm", "S m-1", "log(σ / S cm-1)").
7. **raw_temperature**: Temperature value only (e.g. "25", "298", "2.5" for Arrhenius 1000/T).
8. **raw_temperature_unit**: e.g. "Celsius", "K", "1000/T (K-1)".
9. **source**:
   - "text" = measured by THIS paper's authors in body text
   - "cited_text" = values from other works (with reference numbers like [12])
   - "figure" = extracted from a figure in THIS paper
   - "cited_figure" = figure data attributed to other works
   - "table" = extracted from a table in THIS paper
   - "cited_table" = table data from other works
10. **source_detail**: Which section/figure/table (e.g. "Results Section", "Fig. 3", "Table 2").
11. **processing_method**: Synthesis method if mentioned (e.g. "solution casting", "hot pressing").
12. **aging_time**: Duration if time-series data (e.g. "5 days"). Null otherwise.
13. **measurement_condition**: Special conditions (e.g. "after 100 cycles", "in N2"). Null otherwise.
14. **confidence**: "high" if explicitly stated, "medium" if read from a figure, "low" if inferred/estimated.
15. **warnings**: List of warning strings (e.g. "range_upper_bound", "approximate_from_figure").
16. **exclude_from_ml**: true if this is cited data from introduction/review sections, false otherwise.

IMPORTANT RULES:
- Extract from ALL sources: body text, tables, AND figures (read data points from plots).
- For Arrhenius plots: extract representative points (endpoints + key temperatures like 25°C, 60°C if shown).
- For conductivity RANGES: extract both bounds as separate measurements with "range_upper_bound"/"range_lower_bound" warnings.
- For time-series/aging data: extract EACH time point as a separate measurement.
- For multi-condition samples: extract EACH condition separately.
- Distinguish measured vs cited data carefully — values with reference numbers [X] are almost always cited.
- Do NOT extract activation energy values (eV, kJ/mol) as conductivity.
- Do NOT fabricate data. Only extract what is explicitly present in the paper."""

# ── Core extraction ───────────────────────────────────────────────────────────

def extract_from_pdf(pdf_path: Path, model_name: str = "gemini-2.5-flash",
                     thinking_level: Optional[str] = None) -> dict:
    """Upload PDF to Gemini and extract all ionic conductivity measurements."""
    gemini3 = is_gemini3(model_name)

    # Gemini 3 requires v1alpha for media_resolution support
    client = genai.Client(http_options={"api_version": "v1alpha"}) if gemini3 else genai.Client()
    start = time.time()
    uploaded = None

    # Build PDF part
    if gemini3:
        # Inline data with media_resolution_medium (required for PDFs on Gemini 3)
        print(f"Reading {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB) ...")
        pdf_part = types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=pdf_path.read_bytes(),
            ),
            media_resolution={"level": "media_resolution_medium"},
        )
    else:
        # File upload (Gemini 2.x)
        print(f"Uploading {pdf_path.name} ...")
        uploaded = client.files.upload(file=pdf_path)
        pdf_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type="application/pdf")

    # Build generation config
    config_kwargs = dict(
        response_mime_type="application/json",
        response_json_schema=ExtractionResult.model_json_schema(),
        max_output_tokens=65536,
    )
    if gemini3:
        level = thinking_level or GEMINI3_MODELS[model_name]["default_thinking"]
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        # Do NOT set temperature for Gemini 3 (must stay at default 1.0)
        print(f"Extracting with {model_name} (thinking={level}) ...")
    else:
        config_kwargs["temperature"] = 0.2
        print(f"Extracting with {model_name} ...")

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[pdf_part, types.Part.from_text(text=EXTRACTION_PROMPT)],
            )
        ],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Parse & post-process
    result = ExtractionResult.model_validate_json(response.text)
    for m in result.measurements:
        normalize_units(m)
        if not m.canonical_formula:
            m.canonical_formula = m.raw_composition
        if m.normalized_temperature_c is None and m.raw_conductivity:
            m.normalized_temperature_c = 25.0
            m.warnings.append("assumed_room_temperature")

    elapsed = time.time() - start

    usage = response.usage_metadata
    in_tok  = usage.prompt_token_count or 0
    out_tok = (usage.total_token_count or 0) - in_tok
    cost    = estimate_cost(model_name, in_tok, out_tok)

    # Clean up uploaded file (Gemini 2.x only)
    if uploaded:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    return {
        "doc_name": pdf_path.stem,
        "extraction_version": "pdf_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(elapsed, 2),
        "config": {
            "model": model_name,
            "thinking_level": thinking_level or (GEMINI3_MODELS[model_name]["default_thinking"] if gemini3 else None),
        },
        "token_usage": {
            "input": in_tok,
            "output": out_tok,
            "estimated_cost_usd": round(cost, 5),
        },
        "material_count": len(result.measurements),
        "measurements": [m.model_dump() for m in result.measurements],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract ionic conductivity data from PDF papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Models:
  Gemini 2.5: gemini-2.5-flash (default), gemini-2.5-flash-lite, gemini-2.5-pro
  Gemini 3:   gemini-3-flash-preview, gemini-3.1-flash-lite-preview, gemini-3.1-pro-preview

Thinking levels (Gemini 3 only): minimal | low | medium | high"""
    )
    parser.add_argument("input", type=Path, help="PDF file or directory of PDFs (with --batch)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--thinking-level", default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Thinking depth for Gemini 3 models (ignored for Gemini 2.x)")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    def save_and_report(result: dict, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        cost = result["token_usage"]["estimated_cost_usd"]
        print(f"  -> {result['material_count']} measurements | "
              f"{result['token_usage']['input']:,} in / {result['token_usage']['output']:,} out tokens | "
              f"${cost:.4f} | {result['execution_time_seconds']}s")
        print(f"  -> Saved: {out_path}")

    if args.batch:
        pdfs = sorted(args.input.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {args.input}")
            return
        print(f"Found {len(pdfs)} PDFs — model: {args.model}")
        for pdf in pdfs:
            out_dir = args.output_dir or pdf.parent
            out_path = out_dir / f"{pdf.stem}_extracted.json"
            if out_path.exists():
                print(f"  Skipping {pdf.name} (output exists)")
                continue
            print(f"\n[{pdf.name}]")
            try:
                result = extract_from_pdf(pdf, args.model, args.thinking_level)
                save_and_report(result, out_path)
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        if not args.input.is_file():
            print(f"Error: {args.input} is not a file. Use --batch for directories.")
            return
        result = extract_from_pdf(args.input, args.model, args.thinking_level)
        out_dir = args.output_dir or args.input.parent
        out_path = out_dir / f"{args.input.stem}_extracted.json"
        save_and_report(result, out_path)


if __name__ == "__main__":
    main()
