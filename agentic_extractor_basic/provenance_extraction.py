"""
Per-System Process Provenance Extractor
========================================
Two-call approach per paper:

  Call 1 — Grouping (cheap, no paper text):
    Input:  m0..mN measurement index labels + raw_composition strings
            + pre-extracted paper context summary
    Output: list of material systems, each with assigned measurement indices

  Call 2+ — Extraction (one per system, paper markdown cached when >1 system):
    Input:  paper markdown (cached or inline) + system context
    Output: 3-phase narrative paragraphs for that system

Caching: paper markdown is cached when >1 system is found, reused across
all extraction calls, renewed via heartbeat, deleted on completion.

Output per paper: process_paragraphs_v2.json

Usage:
    mamba activate pokeagent
    python data_extractor/03.2.extract_per_system_provenance.py
    python data_extractor/03.2.extract_per_system_provenance.py --paper "My Paper Folder Name"
    python data_extractor/03.2.extract_per_system_provenance.py --all
    python data_extractor/03.2.extract_per_system_provenance.py --model gemini-2.5-flash --force
    python data_extractor/03.2.extract_per_system_provenance.py --no-llm  # cheap fallback: truncated context for grouping
"""

import os
import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

PAPERS_DIR = Path("./output/downselectedpapers_jiyoung")
DEFAULT_MODEL = "gemini-2.5-flash"
CACHE_PREFIX = "proc-sys-"

DEFAULT_PAPER = (
    "Characterization of the interfacial Li-ion exchange process in a "
    "ceramic–polymer composite by solid state NMR"
)


# ============================================================================
# Schemas
# ============================================================================

class MeasurementSystem(BaseModel):
    system_id: str = Field(
        ...,
        description=(
            "Short descriptive identifier for this material system, "
            "e.g. 'LLZO-NW-composite' or 'PEO-baseline' or 'cited-Zheng2017'"
        ),
    )
    measurement_indices: list[int] = Field(
        ...,
        description=(
            "Integer indices of measurements belonging to this system "
            "(matching the mN labels in the input, e.g. [0, 1, 3]). "
            "Never use composition strings here."
        ),
    )
    composition_summary: str = Field(
        ...,
        description=(
            "Brief human-readable description of this system, "
            "e.g. 'PEO-LiTFSI with LLZO nanofibers at 2–74 wt%'"
        ),
    )


class GroupingResult(BaseModel):
    systems: list[MeasurementSystem] = Field(
        ...,
        description="All distinct material systems found in this paper.",
    )
    is_review_paper: bool = Field(
        ...,
        description=(
            "True if this paper primarily aggregates results from multiple cited studies "
            "rather than reporting its own synthesis."
        ),
    )


class ProcessParagraphs(BaseModel):
    precursor_treatment: str = Field(
        ...,
        description=(
            "Paragraph describing Phase 1: pre-treatment of raw precursor materials "
            "before synthesis begins (drying, sourcing commercial reagents, etc.). "
            "Write 'Not described.' if absent."
        ),
    )
    material_synthesis: str = Field(
        ...,
        description=(
            "Paragraph describing Phase 2: synthesis of individual pure components — "
            "ceramic filler (sol-gel, electrospinning, sintering, ball milling) "
            "and polymer solution (dissolving PEO/PAN + salt). "
            "Write 'Not described.' if absent."
        ),
    )
    composite_manufacturing: str = Field(
        ...,
        description=(
            "Paragraph describing Phase 3: assembly of the final composite — "
            "mixing/dispersing ceramic, casting, drying, pressing, annealing. "
            "Write 'Not described.' if absent."
        ),
    )
    full_process: str = Field(
        ...,
        description=(
            "Single cohesive paragraph summarizing the complete fabrication process "
            "from raw materials to final membrane/pellet, covering all three phases in sequence."
        ),
    )
    completeness: str = Field(
        ...,
        description="One of: 'full' (all phases described), 'partial' (some steps missing or cited), 'minimal'.",
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Process details not found in the paper, e.g. ['sintering temperature', 'ball milling duration'].",
    )
    cited_references: list[str] = Field(
        default_factory=list,
        description="References cited for missing steps, e.g. ['[14] J. Zheng et al. — LLZO synthesis procedure'].",
    )


# ============================================================================
# Caching helpers (mirrors 02.extract_provenance pattern)
# ============================================================================

async def cache_heartbeat(
    client, cache_name: str, stop_event: asyncio.Event, interval: int = 120, ttl: int = 600
):
    """Periodically renew cache TTL so it doesn't expire mid-run."""
    while not stop_event.is_set():
        try:
            client.caches.update(
                name=cache_name,
                config=types.UpdateCachedContentConfig(ttl=f"{ttl}s"),
            )
        except Exception as exc:
            print(f"  ⚠️  Heartbeat failed for {cache_name}: {exc}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


def get_or_create_cache(
    client, model_name: str, paper_name: str, md_content: str
) -> Optional[object]:
    display_name = f"{CACHE_PREFIX}{paper_name[:47]}"

    # Reuse existing cache if present
    try:
        for cached in client.caches.list():
            if cached.display_name == display_name:
                print(f"  ✅ Reusing existing cache: {cached.name}")
                return cached
    except Exception as exc:
        print(f"  ⚠️  Could not list caches: {exc}")

    # Create new
    print(f"  🔄 Creating cache ({len(md_content):,} chars)...")
    try:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                display_name=display_name,
                system_instruction=(
                    "You are an expert in solid-state battery materials synthesis. "
                    "Extract fabrication process details as requested, using only "
                    "information stated in this paper."
                ),
                contents=[md_content],
                ttl="600s",
            ),
        )
        print(f"  ✅ Cache created: {cache.name}")
        return cache
    except Exception as exc:
        print(f"  ❌ Cache creation failed: {exc}")
        return None


# ============================================================================
# Call 1 — Grouping
# ============================================================================

GROUPING_SYSTEM_PROMPT = (
    "You are an expert in solid-state battery materials science. "
    "Group measurement entries into distinct material systems based on their "
    "fabrication route, not their measurement conditions or filler loading."
)


def build_grouping_prompt(
    measurements: list, paper_context: dict, md_content: Optional[str] = None
) -> str:
    lines = []
    for i, m in enumerate(measurements):
        source = m.get("source") or ""
        if source.startswith("cited"):
            continue
        comp = m.get("raw_composition") or "Unknown"
        mat_class = m.get("material_class") or ""
        lines.append(f"m{i}: [{mat_class}] {comp}")

    measurements_block = "\n".join(lines) if lines else "(no non-cited measurements)"

    ctx_parts = []
    for key in ["material_systems_overview", "experimental_procedure_summary"]:
        val = paper_context.get(key)
        if val:
            trunc = None if md_content else 600
            ctx_parts.append(f"[{key}]: {val if trunc is None else val[:trunc]}")
    context_block = "\n".join(ctx_parts) if ctx_parts else "None available."

    paper_block = (
        f"\n=== Full Paper Text ===\n{md_content}\n=== End Paper Text ===\n"
        if md_content
        else ""
    )

    return f"""Group the following measurements from a polymer-ceramic composite electrolyte paper
into distinct material systems — systems that share the same fabrication route.

=== Measurements ===
{measurements_block}
=== End Measurements ===
{paper_block}
=== Paper context (summary) ===
{context_block}
=== End context ===

Rules:
- Measurements that differ only in filler loading (e.g., 2 wt% vs 10 wt% of the same ceramic) → same system
- Measurements with fundamentally different chemistries (e.g., PAN-based vs PEO-based matrix) → separate systems
- A pure polymer baseline (0 wt% filler) may be its own system if the ceramic synthesis is absent,
  or grouped with the composite series if the rest of the process is shared — use your judgment
- For literature review papers: create one system per cited study
- Use only integer indices (0, 1, 2, ...) in measurement_indices — never composition strings
- Every non-cited measurement index must appear in exactly one system
- Cited measurements (marked with [cited_*] in the source, not listed above) are excluded
"""


async def call_grouping(
    client,
    model_name: str,
    measurements: list,
    paper_context: dict,
    md_content: Optional[str] = None,
) -> Optional[GroupingResult]:
    prompt = build_grouping_prompt(measurements, paper_context, md_content)
    config = types.GenerateContentConfig(
        system_instruction=GROUPING_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GroupingResult,
        temperature=0.0,
    )
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        raw = json.loads(response.text)
        return GroupingResult.model_validate(raw)
    except Exception as exc:
        print(f"  ❌ Grouping call failed: {exc}")
        return None


# ============================================================================
# Call 2+ — Per-system extraction
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in solid-state battery materials synthesis. "
    "Extract and summarize the fabrication process for the specified material system "
    "as clear, concise narrative paragraphs."
)


def build_extraction_prompt(
    system: MeasurementSystem,
    measurements: list,
    paper_context: dict,
    md_content: Optional[str],  # None when using cache
) -> str:
    comp_lines = []
    for idx in system.measurement_indices:
        if idx < len(measurements):
            comp = measurements[idx].get("raw_composition") or "Unknown"
            comp_lines.append(f"  m{idx}: {comp}")
    compositions_block = "\n".join(comp_lines) if comp_lines else "  (see composition_summary)"

    ctx_parts = []
    for key in ["experimental_procedure_summary", "nomenclature_key", "material_systems_overview"]:
        val = paper_context.get(key)
        if val:
            ctx_parts.append(f"[{key}]: {val}")
    context_block = "\n".join(ctx_parts) if ctx_parts else "None available."

    # Paper text only injected inline when not using cache
    paper_block = (
        f"\n=== Full Paper Text ===\n{md_content}\n=== End Paper Text ===\n"
        if md_content
        else ""
    )

    return f"""Extract the fabrication process for the following material system.
Focus ONLY on this system. If other material systems are described in the paper, ignore them.

=== Target System ===
system_id: {system.system_id}
description: {system.composition_summary}
measurements in this system:
{compositions_block}
=== End Target System ===
{paper_block}
=== Pre-extracted paper context ===
{context_block}
=== End context ===

Organize the fabrication process into exactly three phases:

PHASE 1 — Precursor Treatment:
Pre-treatment of raw precursor materials before synthesis begins. Drying reagents,
dissolving precursor salts, sourcing commercial reagents. Many papers skip this phase.

PHASE 2 — Material Synthesis:
Synthesis of individual pure components:
(a) Ceramic synthesis (sol-gel / chelate-gel / electrospinning → calcination → sintering → milling)
(b) Polymer solution preparation (dissolving PEO/PAN + salt in solvent)
If the ceramic was purchased commercially with no synthesis described, write "Not described."

PHASE 3 — Composite Manufacturing:
Assembly of the final composite: mixing/dispersing ceramic into polymer solution,
casting onto substrate, solvent evaporation/drying, pressing, annealing, post-processing.

Rules:
- Use ONLY information stated in the paper. Do NOT invent steps or values.
- If a phase is absent, write "Not described."
- If a step refers to another publication (e.g., "prepared according to [14]"), do NOT invent those
  steps — note the reference in cited_references with its number and bibliographic text.
- Write each phase as a flowing paragraph, not a bullet list.
- Write a single full_process paragraph combining all phases in sequence.
"""


async def call_extraction(
    client,
    model_name: str,
    cache_name: Optional[str],
    system: MeasurementSystem,
    measurements: list,
    paper_context: dict,
    md_content: Optional[str],
) -> Optional[ProcessParagraphs]:
    prompt = build_extraction_prompt(system, measurements, paper_context, md_content)

    if cache_name:
        config = types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_schema=ProcessParagraphs,
            temperature=0.1,
        )
    else:
        config = types.GenerateContentConfig(
            system_instruction=EXTRACTION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ProcessParagraphs,
            temperature=0.1,
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw = json.loads(response.text)
            return ProcessParagraphs.model_validate(raw)
        except Exception as exc:
            err = str(exc)
            if any(c in err for c in ["429", "500", "503"]) and attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"    ⚠️  Attempt {attempt + 1} failed, retry in {wait}s: {exc}")
                await asyncio.sleep(wait)
                continue
            print(f"    ❌ Extraction failed for '{system.system_id}': {exc}")
            return None


# ============================================================================
# Paper loading
# ============================================================================

def load_paper(
    paper_name: str, papers_dir: Path
) -> tuple[str, dict, list]:
    """Returns (markdown_text, paper_context, measurements)."""
    paper_dir = papers_dir / paper_name

    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md file in {paper_dir}")
    with open(md_files[0], encoding="utf-8") as f:
        md_text = f.read()

    paper_context: dict = {}
    measurements: list = []

    ctx_path = paper_dir / "t0_paper_context.json"
    v8_path = paper_dir / "robust_results_v8.json"

    if ctx_path.exists():
        with open(ctx_path, encoding="utf-8") as f:
            paper_context = json.load(f)

    if v8_path.exists():
        with open(v8_path, encoding="utf-8") as f:
            data = json.load(f)
        if not paper_context:
            paper_context = data.get("paper_context", {})
        measurements = data.get("measurements", [])

    return md_text, paper_context, measurements


# ============================================================================
# Per-paper orchestration
# ============================================================================

async def process_paper(
    client,
    paper_name: str,
    papers_dir: Path,
    model_name: str,
    force: bool,
    llm_grouping: bool = False,
) -> dict:
    paper_dir = papers_dir / paper_name
    output_fname = "process_paragraphs_v2_llm_grouping.json" if llm_grouping else "process_paragraphs_v2.json"
    output_path = paper_dir / output_fname

    if output_path.exists() and not force:
        print(f"  ⏭️  Already processed: {output_path.name}  (use --force to re-run)")
        return {"paper": paper_name, "status": "skipped"}

    print(f"\n{'='*70}")
    print(f"📄 {paper_name}")
    print(f"   Model: {model_name}")
    print(f"{'='*70}")

    try:
        md_text, paper_context, measurements = load_paper(paper_name, papers_dir)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return {"paper": paper_name, "status": "no_md"}

    non_cited = [m for m in measurements if not (m.get("source") or "").startswith("cited")]
    print(f"  Measurements: {len(measurements)} total | {len(non_cited)} non-cited")

    if not non_cited:
        print("  ⚠️  No non-cited measurements — nothing to group")
        return {"paper": paper_name, "status": "no_measurements"}

    # ----------------------------------------------------------------
    # Call 1 — Grouping
    # ----------------------------------------------------------------
    grouping_mode = "full paper text" if llm_grouping else "context summary only"
    print(f"  🔍 Call 1: Grouping {len(non_cited)} measurements into material systems ({grouping_mode})...")
    grouping = await call_grouping(
        client, model_name, measurements, paper_context,
        md_content=md_text if llm_grouping else None,
    )

    if not grouping or not grouping.systems:
        print("  ❌ Grouping returned no systems")
        return {"paper": paper_name, "status": "grouping_failed"}

    systems = grouping.systems
    review_tag = " [review paper]" if grouping.is_review_paper else ""
    print(f"  ✅ {len(systems)} system(s) found{review_tag}:")
    for s in systems:
        idx_str = ", ".join(f"m{i}" for i in s.measurement_indices[:6])
        if len(s.measurement_indices) > 6:
            idx_str += f", ... ({len(s.measurement_indices)} total)"
        print(f"     • {s.system_id}: {idx_str}")
        print(f"       {s.composition_summary[:90]}")

    # ----------------------------------------------------------------
    # Cache: create only when >1 system (otherwise single call, no benefit)
    # ----------------------------------------------------------------
    use_cache = len(systems) > 1
    cache = None
    heartbeat_task = None
    stop_heartbeat = asyncio.Event()

    if use_cache:
        cache = get_or_create_cache(client, model_name, paper_name, md_text)
        if cache:
            heartbeat_task = asyncio.create_task(
                cache_heartbeat(client, cache.name, stop_heartbeat)
            )
        else:
            print("  ⚠️  Cache unavailable — falling back to inline paper text per call")

    # ----------------------------------------------------------------
    # Call 2+ — Extract process per system (run concurrently)
    # ----------------------------------------------------------------
    print(f"  📝 Call 2+: Extracting process for {len(systems)} system(s)...")
    try:
        tasks = [
            call_extraction(
                client=client,
                model_name=model_name,
                cache_name=cache.name if cache else None,
                system=s,
                measurements=measurements,
                paper_context=paper_context,
                md_content=md_text if not cache else None,
            )
            for s in systems
        ]
        results = await asyncio.gather(*tasks)
    finally:
        if heartbeat_task:
            stop_heartbeat.set()
            await heartbeat_task

    # Clean up cache immediately after all extractions complete
    if cache:
        print(f"  🧹 Deleting cache: {cache.name}")
        try:
            client.caches.delete(name=cache.name)
        except Exception as exc:
            print(f"  ⚠️  Cache deletion failed: {exc}")

    # ----------------------------------------------------------------
    # Build output
    # ----------------------------------------------------------------
    output_systems = []
    for s, result in zip(systems, results):
        entry: dict = {
            "system_id": s.system_id,
            "measurement_indices": s.measurement_indices,
            "composition_summary": s.composition_summary,
        }
        if result:
            entry.update({
                "completeness": result.completeness,
                "missing_info": result.missing_info,
                "cited_references": result.cited_references,
                "phases": {
                    "precursor_treatment": result.precursor_treatment,
                    "material_synthesis": result.material_synthesis,
                    "composite_manufacturing": result.composite_manufacturing,
                },
                "full_process": result.full_process,
            })
            print(f"     ✅ {s.system_id}: {result.completeness}")
        else:
            entry["extraction_failed"] = True
            print(f"     ❌ {s.system_id}: extraction failed")
        output_systems.append(entry)

    output = {
        "paper": paper_name,
        "model": model_name,
        "created_at": datetime.now().isoformat(),
        "is_review_paper": grouping.is_review_paper,
        "is_single_system": len(systems) == 1,
        "material_systems": output_systems,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for s in output_systems if not s.get("extraction_failed"))
    print(f"  💾 Saved: {output_path.name}  ({n_ok}/{len(systems)} systems extracted)")

    return {
        "paper": paper_name,
        "status": "success",
        "systems": len(systems),
        "is_review_paper": grouping.is_review_paper,
    }


# ============================================================================
# Entry point
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Extract per-system fabrication process from a paper."
    )
    parser.add_argument(
        "--paper",
        default=DEFAULT_PAPER,
        help="Exact paper folder name inside papers-dir",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help=f"Root directory containing paper folders (default: {PAPERS_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output already exists",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_papers",
        help="Process all paper folders in papers-dir",
    )
    parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="llm_grouping",
        help=(
            "Use only the truncated context summary for grouping (Call 1) instead of "
            "the full paper markdown. Cheaper but less accurate for complex papers; "
            "output saved as process_paragraphs_v2.json."
        ),
    )
    parser.set_defaults(llm_grouping=True)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return

    client = genai.Client(api_key=api_key)

    if args.all_papers:
        paper_names = [
            d.name
            for d in sorted(args.papers_dir.iterdir())
            if d.is_dir() and list(d.glob("*.md"))
        ]
        print(f"Found {len(paper_names)} papers to process.")
        summary = []
        for name in paper_names:
            result = await process_paper(client, name, args.papers_dir, args.model, args.force, args.llm_grouping)
            summary.append(result)

        print(f"\n{'='*70}")
        print("Run summary:")
        for r in summary:
            status = r.get("status")
            systems = r.get("systems", "-")
            review = " [review]" if r.get("is_review_paper") else ""
            print(f"  {r['paper'][:60]:<60} {status} ({systems} systems){review}")
    else:
        await process_paper(client, args.paper, args.papers_dir, args.model, args.force, args.llm_grouping)


if __name__ == "__main__":
    asyncio.run(main())
