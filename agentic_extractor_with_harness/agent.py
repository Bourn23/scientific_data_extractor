"""
PageIndex extraction entry point.

Runs the extraction pipeline directly (no LLM orchestration overhead), saves
results to robust_results_v8.json, then hands off to the verifier agent to
recover failed extractions and do a full PDF sweep.

Usage:
    python agent.py <path_to_paper.md> [path_to_paper.pdf]

The PDF argument is optional — if omitted the script looks for a .pdf file
in the same directory as the markdown file.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Post-processing helpers (mirrors logic from extraction_logic.main)
# =============================================================================

def _is_missing(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "null", "none", "n/a", "unknown", "not specified"}


def _has_conductivity(m) -> bool:
    if m.normalized_conductivity is not None:
        return True
    raw = str(m.raw_conductivity).strip().lower() if m.raw_conductivity is not None else ""
    return raw not in {"", "null", "none", "n/a", "unknown", "not specified"}


def _apply_missing_temperature_assumptions(measurements, assumed_rt_c: float = 25.0) -> int:
    assumed_count = 0
    warning_msg = f"Assumed room temperature ({assumed_rt_c:.0f} °C) because temperature was missing."
    for m in measurements:
        if m.normalized_temperature_c is not None:
            continue
        if not _has_conductivity(m):
            continue
        m.normalized_temperature_c = assumed_rt_c
        if _is_missing(m.raw_temperature):
            m.raw_temperature = str(int(assumed_rt_c))
        if _is_missing(m.raw_temperature_unit):
            m.raw_temperature_unit = "room temperature"
        if m.warnings is None:
            m.warnings = []
        if warning_msg not in m.warnings:
            m.warnings.append(warning_msg)
        assumed_count += 1
    return assumed_count


# =============================================================================
# Main pipeline + verifier flow
# =============================================================================

async def extract_and_verify(markdown_file_path: str, pdf_path: str = None) -> None:
    from extraction_logic import run_pipeline, VISION_MODEL, TEXT_MODEL, tracker
    from verifier_agent import run_verifier

    md_path = Path(markdown_file_path).resolve()
    asset_dir = md_path.parent

    print(f"\n{'=' * 60}")
    print("PageIndex Extraction Pipeline")
    print(f"Document : {md_path.name}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Phase 1: Extraction
    # ------------------------------------------------------------------
    print("Phase 1: Running extraction pipeline...")
    t0 = time.time()

    measurements, material_defs, stats, is_review_article, paper_context = await run_pipeline(
        md_path, asset_dir, model="gemini-3-flash-preview"
    )

    assumed = _apply_missing_temperature_assumptions(measurements)
    if assumed:
        print(f"   Applied RT assumption to {assumed} measurement(s) with missing temperature.")

    elapsed = round(time.time() - t0, 2)

    out_path = md_path.parent / "robust_results_v8.json"
    output_data = {
        "doc_name": md_path.stem,
        "extraction_version": "v8",
        "is_review_article": is_review_article,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": elapsed,
        "config": {
            "vision_model": VISION_MODEL,
            "text_model": TEXT_MODEL,
            "normalization_engine": "v2_hybrid_llm_python",
        },
        "cost_summary": {
            "total_input_tokens": tracker.total_input_tokens,
            "total_output_tokens": tracker.total_output_tokens,
            "total_cost_usd": round(tracker.total_cost_usd, 4),
            "call_counts": tracker.call_counts,
        },
        "extraction_stats": stats,
        "paper_context": paper_context or {},
        "material_count": len(measurements),
        "measurements": [m.model_dump() for m in measurements],
        "material_definitions": material_defs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPhase 1 complete: {len(measurements)} measurements saved to {out_path}")
    tracker.print_summary()

    # ------------------------------------------------------------------
    # Phase 2: Verifier
    # ------------------------------------------------------------------
    log_path = md_path.parent / "extraction_log.jsonl"
    if not log_path.exists():
        print("\nNo extraction log found — skipping verifier.")
        return

    # Resolve PDF path
    if not pdf_path:
        candidates = list(md_path.parent.glob("*.pdf"))
        if candidates:
            pdf_path = str(candidates[0])
            print(f"\nFound PDF: {pdf_path}")
        else:
            print("\nNo PDF found alongside markdown — verifier will skip PDF verification.")
            pdf_path = ""

    print(f"\n{'=' * 60}")
    print("Phase 2: Running verifier agent...")
    print(f"{'=' * 60}")

    report = await run_verifier(
        log_path=str(log_path),
        results_json_path=str(out_path),
        pdf_path=pdf_path,
    )

    print(f"\n{'=' * 60}")
    print("VERIFIER REPORT")
    print(f"{'=' * 60}")
    print(report)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_paper.md> [path_to_paper.pdf]")
        print("Example: python agent.py ./papers/composite_electrolyte_2025.md")
        sys.exit(1)

    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None

    asyncio.run(extract_and_verify(md_file, pdf_file))
