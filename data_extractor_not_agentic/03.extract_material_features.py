"""
03.extract_material_features.py
===============================
Extracts material identity features (polymer_class, ceramic_type, salt_type,
filler_loading_wt_pct, filler_morphology) from per-system provenance output
produced by 03.2.extract_per_system_provenance.py.

Reads:
  - robust_results_v8.json         (measurements + paper_context)
  - process_paragraphs_v2_llm_grouping.json  (03.2 provenance output)

Outputs per paper:
  - T0_structured_features.csv     (inside the paper folder)

Consolidated output:
  - T0_structured_features.csv     (in OUTPUT_DIR, all papers merged)

Usage:
    mamba activate pokeagent
    python data_extractor/03.extract_material_features.py --paper "My Paper Folder Name"
    python data_extractor/03.extract_material_features.py --all
    python data_extractor/03.extract_material_features.py --all --force
"""

import os
import json
import asyncio
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

PAPERS_DIR = Path("./output/downselectedpapers_jiyoung")
OUTPUT_DIR = Path("./actionable_analysis/material_features_output")
DEFAULT_MODEL = "gemini-2.5-flash"
CONCURRENCY_LIMIT = 4

PROVENANCE_FILENAME = "process_paragraphs_v2_llm_grouping.json"

DEFAULT_PAPER = (
    "Characterization of the interfacial Li-ion exchange process in a "
    "ceramic–polymer composite by solid state NMR"
)

# ============================================================================
# Models
# ============================================================================

class CompositionFeatures(BaseModel):
    raw_composition: str = Field(
        ..., description="The original composition string from the paper"
    )
    polymer_class: Literal[
        'PEO', 'PVDF', 'PAN', 'PMMA', 'PPC', 'PEC', 'PVC', 'PTMC', 'other', 'none'
    ] = Field(..., description="The primary polymer matrix")
    salt_type: Literal[
        'LiTFSI', 'LiFSI', 'LiClO4', 'LiPF6', 'LiBF4', 'LiDFOB', 'other', 'none'
    ] = Field(..., description="Lithium salt identity")
    ceramic_type: Literal[
        'LLZO', 'LATP', 'LLTO', 'LAGP', 'SiO2', 'TiO2', 'Al2O3', 'BaTiO3',
        'MOF', 'COF', 'other', 'none'
    ] = Field(..., description="Primary ceramic filler identity")
    filler_loading_wt_pct: Optional[float] = Field(
        default=None, description="Weight percent of ceramic filler (0-100)"
    )
    filler_morphology: Literal[
        'nanoparticle', 'nanowire', 'nanosheet', 'nanofiber',
        'framework_3D', 'microsphere', 'mixed', 'unknown'
    ] = Field(..., description="Filler morphology/dimensionality")


class SystemMaterialExtraction(BaseModel):
    system_id: str = Field(..., description="The material system identifier")
    compositions: List[CompositionFeatures] = Field(
        ..., description="Extracted features for each composition in the system"
    )


# ============================================================================
# Prompt Building
# ============================================================================

def build_system_prompt(
    paper_name: str,
    system: dict,
    compositions: list[str],
    paper_context: dict,
) -> str:
    """Build LLM prompt for extracting material features from a 03.2 system."""

    # Paper context
    ctx_lines = []
    for field in [
        "experimental_procedure_summary",
        "nomenclature_key",
        "material_systems_overview",
    ]:
        if val := paper_context.get(field):
            ctx_lines.append(f"[{field}]: {val}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "None available."

    # Process narrative from 03.2 phases
    phases = system.get("phases", {})
    phase_lines = []
    for phase_name in ["precursor_treatment", "material_synthesis", "composite_manufacturing"]:
        text = phases.get(phase_name, "Not described.")
        phase_lines.append(f"[{phase_name}]: {text}")
    phases_block = "\n\n".join(phase_lines)

    full_process = system.get("full_process", "")

    # Compositions to extract
    comps_block = "\n".join([f"- {c}" for c in compositions])

    return f"""You are an expert materials scientist. Parse the material identity and morphology features for the specific sample compositions provided.

Paper: {paper_name}

=== Paper Overview Context ===
{context_block}

=== System: {system.get('system_id', 'unknown')} ===
Summary: {system.get('composition_summary', '')}

=== Fabrication Process (from provenance extraction) ===
{phases_block}

Full process summary:
{full_process}

=== Compositions to Extract ===
We need to extract structured data for the following specific compositions reported in this system:
{comps_block}

Instructions:
1. For EACH composition listed above, create a CompositionFeatures object.
2. Use the Paper Overview Context AND the Fabrication Process to determine `polymer_class`, `salt_type`, `ceramic_type`, `filler_loading_wt_pct`, and `filler_morphology`.
3. The fabrication process narrative is especially helpful for determining filler loadings (e.g., '40 wt% LLZO') and morphology (e.g., 'nanowires', 'nanofibers').
4. If a value cannot be found or inferred, use 'none', 'unknown', or null as appropriate.
5. Return the result using system_id: {system.get('system_id', 'unknown')}
"""


def build_fallback_prompt(
    paper_name: str,
    compositions: list[str],
    paper_context: dict,
) -> str:
    """Prompt for measurements not covered by any 03.2 system (cited, etc.)."""

    ctx_lines = []
    for field in [
        "experimental_procedure_summary",
        "nomenclature_key",
        "material_systems_overview",
    ]:
        if val := paper_context.get(field):
            ctx_lines.append(f"[{field}]: {val}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "None available."

    comps_block = "\n".join([f"- {c}" for c in compositions])

    return f"""You are an expert materials scientist. Parse the material identity and morphology features for the specific sample compositions provided.

Paper: {paper_name}

=== Paper Overview Context ===
{context_block}

=== Compositions to Extract ===
{comps_block}

Instructions:
1. For EACH composition listed above, create a CompositionFeatures object.
2. Use the composition string and paper context to determine `polymer_class`, `salt_type`, `ceramic_type`, `filler_loading_wt_pct`, and `filler_morphology`.
3. Many of these may be cited literature values with generic descriptions. Do your best.
4. If a value cannot be found or inferred, use 'none', 'unknown', or null as appropriate.
5. Return the result using system_id: "uncovered"
"""


# ============================================================================
# LLM Extraction
# ============================================================================

async def extract_features_for_system(
    client, sem, model_name: str,
    paper_name: str,
    system: dict,
    compositions: list[str],
    paper_context: dict,
) -> Optional[SystemMaterialExtraction]:
    """Call LLM to extract material features for one 03.2 material system."""
    prompt = build_system_prompt(paper_name, system, compositions, paper_context)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SystemMaterialExtraction,
        temperature=0.0,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with sem:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            raw = json.loads(response.text)
            return SystemMaterialExtraction.model_validate(raw)
        except Exception as e:
            err_str = str(e)
            transient = any(code in err_str for code in ["403", "429", "500", "503"])
            if transient and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"    ⚠️  Retry {attempt + 1} for '{system.get('system_id')}' in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                continue
            print(f"    ❌ Failed for system '{system.get('system_id')}': {e}")
            return None


async def extract_features_fallback(
    client, sem, model_name: str,
    paper_name: str,
    compositions: list[str],
    paper_context: dict,
) -> Optional[SystemMaterialExtraction]:
    """Extract features for measurements not covered by any 03.2 system."""
    prompt = build_fallback_prompt(paper_name, compositions, paper_context)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SystemMaterialExtraction,
        temperature=0.0,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with sem:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            raw = json.loads(response.text)
            return SystemMaterialExtraction.model_validate(raw)
        except Exception as e:
            err_str = str(e)
            transient = any(code in err_str for code in ["403", "429", "500", "503"])
            if transient and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"    ⚠️  Retry {attempt + 1} for fallback in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                continue
            print(f"    ❌ Fallback extraction failed: {e}")
            return None


# ============================================================================
# Per-Paper Processing
# ============================================================================

MATERIAL_CLASS_MAP = {
    "Composite": "polymer_ceramic_composite",
    "Polymer": "pure_polymer",
    "Ceramic": "pure_ceramic",
}

CSV_FIELDNAMES = [
    "sample_id", "doc_name", "raw_composition", "canonical_formula",
    "conductivity_S_cm", "measurement_temperature_c", "source_type",
    "confidence", "polymer_class", "salt_type", "ceramic_type",
    "filler_loading_wt_pct", "filler_loading_vol_pct", "filler_morphology",
    "material_class",
]


async def process_paper(
    client,
    paper_name: str,
    papers_dir: Path,
    model_name: str,
    sem: asyncio.Semaphore,
    force: bool = False,
) -> Optional[list[dict]]:
    """Process a single paper: read v8 + 03.2 provenance, extract features."""

    paper_dir = papers_dir / paper_name
    output_csv = paper_dir / "T0_structured_features.csv"

    if output_csv.exists() and not force:
        print(f"  ⏭️  Already exists: {output_csv.name}  (use --force to re-run)")
        return None

    # --- Load robust_results_v8.json ---
    v8_file = paper_dir / "robust_results_v8_post_jiyoung_fixes.json"
    if not v8_file.exists():
        v8_file = paper_dir / "robust_results_v8.json"
    if not v8_file.exists():
        print(f"  ⚠️  No V8 measurements in {paper_name}")
        return None

    with open(v8_file, encoding="utf-8") as f:
        v8_data = json.load(f)

    doc_name = v8_data.get("doc_name", paper_name)
    measurements = v8_data.get("measurements", [])
    paper_context = v8_data.get("paper_context", {})

    if not measurements:
        print(f"  ⚠️  No measurements in {paper_name}")
        return None

    # --- Load 03.2 provenance ---
    prov_file = paper_dir / PROVENANCE_FILENAME
    prov_data = {}
    if prov_file.exists():
        with open(prov_file, encoding="utf-8") as f:
            prov_data = json.load(f)

    systems = prov_data.get("material_systems", [])

    print(f"\n{'='*70}")
    print(f"📄 {doc_name}")
    print(f"   {len(measurements)} measurements, {len(systems)} material system(s) from 03.2")
    print(f"{'='*70}")

    # --- Build index: measurement_index → system ---
    idx_to_system: dict[int, dict] = {}
    for sys in systems:
        for idx in sys.get("measurement_indices", []):
            idx_to_system[idx] = sys

    # --- Collect unique compositions per system (preserving order) ---
    system_compositions: dict[str, list[str]] = {}  # system_id → [raw_compositions]
    uncovered_compositions: list[tuple[int, str]] = []  # (index, raw_comp) for orphans

    for i, m in enumerate(measurements):
        raw_comp = m.get("raw_composition", "Unknown")
        if i in idx_to_system:
            sid = idx_to_system[i]["system_id"]
            system_compositions.setdefault(sid, [])
            if raw_comp not in system_compositions[sid]:
                system_compositions[sid].append(raw_comp)
        else:
            uncovered_compositions.append((i, raw_comp))

    # --- Extract features per system (concurrent) ---
    tasks = []
    system_ids_order = []
    input_comp_lists: list[list[str]] = []  # track input order for positional matching

    for sys in systems:
        sid = sys["system_id"]
        comps = system_compositions.get(sid, [])
        if comps:
            print(f"  🔍 System '{sid}': {len(comps)} unique composition(s)")
            tasks.append(
                extract_features_for_system(
                    client, sem, model_name, doc_name, sys, comps, paper_context
                )
            )
            system_ids_order.append(sid)
            input_comp_lists.append(comps)

    # Fallback for uncovered measurements
    uncovered_comps_unique = list(dict.fromkeys(c for _, c in uncovered_compositions))
    if uncovered_comps_unique:
        print(f"  🔍 Uncovered measurements: {len(uncovered_comps_unique)} unique composition(s)")
        tasks.append(
            extract_features_fallback(
                client, sem, model_name, doc_name, uncovered_comps_unique, paper_context
            )
        )
        system_ids_order.append("__uncovered__")
        input_comp_lists.append(uncovered_comps_unique)

    results = await asyncio.gather(*tasks)

    # --- Build composition → features map ---
    # Use both positional matching (primary) and string matching (fallback)
    comp_feature_map: dict[str, dict] = {}
    for sid, result, input_comps in zip(system_ids_order, results, input_comp_lists):
        if result:
            # Positional matching: map input composition[i] → result.compositions[i]
            for j, input_comp in enumerate(input_comps):
                if j < len(result.compositions):
                    clean = input_comp.strip().lower()
                    comp_feature_map[clean] = result.compositions[j].model_dump()

            # Also add string-based matching as fallback for any extras
            for comp in result.compositions:
                clean = comp.raw_composition.strip().lower()
                if clean not in comp_feature_map:
                    comp_feature_map[clean] = comp.model_dump()

            print(f"    ✅ '{sid}': {len(result.compositions)} composition(s) extracted")
        else:
            print(f"    ❌ '{sid}': extraction failed")

    # --- Build flat measurement list ---
    enriched = []
    for i, m in enumerate(measurements):
        raw_comp = m.get("raw_composition", "Unknown")
        clean = raw_comp.strip().lower()
        feats = comp_feature_map.get(clean, {})

        raw_mc = m.get("material_class", "Composite")

        flat = {
            "sample_id": f"{doc_name}__m{i}",
            "doc_name": doc_name,
            "raw_composition": raw_comp,
            "canonical_formula": m.get("canonical_formula"),
            "conductivity_S_cm": m.get("normalized_conductivity"),
            "measurement_temperature_c": m.get("normalized_temperature_c"),
            "source_type": m.get("source"),
            "confidence": m.get("confidence"),
            "polymer_class": feats.get("polymer_class"),
            "salt_type": feats.get("salt_type"),
            "ceramic_type": feats.get("ceramic_type"),
            "filler_loading_wt_pct": feats.get("filler_loading_wt_pct"),
            "filler_loading_vol_pct": None,
            "filler_morphology": feats.get("filler_morphology"),
            "material_class": MATERIAL_CLASS_MAP.get(raw_mc, raw_mc),
        }
        enriched.append(flat)

    # --- Save per-paper CSV ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched)

    n_with_feats = sum(1 for e in enriched if e.get("polymer_class") is not None)
    print(f"  💾 Saved {output_csv.name}: {len(enriched)} rows ({n_with_feats} with features)")

    return enriched


# ============================================================================
# Entry Point
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Extract material features from 03.2 provenance output."
    )
    parser.add_argument(
        "--paper", default=DEFAULT_PAPER,
        help="Exact paper folder name inside papers-dir",
    )
    parser.add_argument(
        "--papers-dir", type=Path, default=PAPERS_DIR,
        help=f"Root directory containing paper folders (default: {PAPERS_DIR})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--force", action="store_true", help="Re-extract even if output exists")
    parser.add_argument(
        "--all", action="store_true", dest="all_papers",
        help="Process all paper folders that have 03.2 provenance output",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return

    client = genai.Client(api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Determine papers to process ---
    if args.all_papers:
        paper_names = [
            d.name
            for d in sorted(args.papers_dir.iterdir())
            if d.is_dir() and (d / PROVENANCE_FILENAME).exists()
        ]
        print(f"Found {len(paper_names)} papers with 03.2 provenance output.")
    else:
        paper_names = [args.paper]

    # --- Process papers ---
    all_measurements = []
    summary = []

    for name in paper_names:
        result = await process_paper(
            client, name, args.papers_dir, args.model, sem, args.force
        )
        if result:
            all_measurements.extend(result)
            summary.append({"paper": name, "n": len(result)})
        else:
            summary.append({"paper": name, "n": 0})

    # --- Save consolidated CSV ---
    if all_measurements:
        csv_path = OUTPUT_DIR / "T0_structured_features.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_measurements)
        print(f"\n💾 Consolidated: {len(all_measurements)} rows → {csv_path}")

    # --- Summary ---
    if len(summary) > 1:
        print(f"\n{'='*70}")
        print("Run summary:")
        for s in summary:
            print(f"  {s['paper'][:55]:<55} {s['n']} rows")


if __name__ == "__main__":
    asyncio.run(main())
