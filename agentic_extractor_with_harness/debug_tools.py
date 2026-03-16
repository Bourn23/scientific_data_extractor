"""
Quick diagnostic: run each tool in isolation on a real paper file.
Usage:
    python debug_tools.py <path_to_paper.md>

Checks in order:
  1. parse_markdown_structure   — does it find sections/images/tables?
  2. retrieve_document_abbreviations — does the LLM extract any mappings?
  3. extract_from_markdown_table — does the first table yield measurements?
  4. QA_process_measurement_batch — does QA run without errors?
"""
import asyncio
import json
import sys
from pathlib import Path

from tools import (
    parse_markdown_structure,
    retrieve_document_abbreviations,
    extract_from_markdown_table,
    QA_process_measurement_batch,
)


async def main(md_path: str):
    resolved = Path(md_path).resolve()
    text = resolved.read_text(encoding="utf-8")
    asset_dir = str(resolved.parent)

    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: parse_markdown_structure")
    print("="*60)
    structure = parse_markdown_structure(text, asset_dir)
    print(f"  Title   : {structure['title']}")
    print(f"  Sections: {len(structure['sections'])}")
    print(f"  Images  : {len(structure['images'])}")
    print(f"  Tables  : {len(structure['tables'])}")
    for s in structure['sections'][:5]:
        print(f"    - [{s['title']}] line {s['line_num']}–{s['end_line_num']}")
    for img in structure['images'][:5]:
        print(f"    - IMG {img['figure_id']}: {img['filename']}  caption={img['caption'][:60]!r}")
    for tbl in structure['tables'][:3]:
        print(f"    - TBL {tbl['tab_id']}: caption={tbl['caption'][:60]!r}")

    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: retrieve_document_abbreviations")
    print("="*60)
    # Feed only intro+methods text to keep it fast
    intro_text = "\n".join(
        s["content_preview"]
        for s in structure["sections"]
        if any(kw in s["title"].lower() for kw in
               ["abstract", "intro", "method", "experimental", "synthesis", "preparation"])
    )
    if not intro_text.strip():
        intro_text = text[:4000]
        print("  (no intro/methods sections found, using first 4000 chars)")

    abbrev_map = await retrieve_document_abbreviations(intro_text)
    print(f"  Found {len(abbrev_map)} abbreviation mappings:")
    for k, v in list(abbrev_map.items())[:10]:
        print(f"    {k!r:20s} → {v!r}")

    # ------------------------------------------------------------------
    if structure["tables"]:
        print("\n" + "="*60)
        print("STEP 3: extract_from_markdown_table (first table)")
        print("="*60)
        first_table = structure["tables"][0]
        measurements = await extract_from_markdown_table(
            first_table["content"], first_table["caption"]
        )
        print(f"  Extracted {len(measurements)} measurements from {first_table['tab_id']}")
        for m in measurements[:5]:
            print(f"    comp={m['raw_composition']!r:30s}  "
                  f"σ={m['raw_conductivity']} {m['raw_conductivity_unit']}  "
                  f"T={m['raw_temperature']} {m['raw_temperature_unit']}")

        # ------------------------------------------------------------------
        if measurements:
            print("\n" + "="*60)
            print("STEP 4: QA_process_measurement_batch")
            print("="*60)
            qa_results = QA_process_measurement_batch(measurements, abbrev_map)
            print(f"  QA processed {len(qa_results)} measurements")
            high = sum(1 for m in qa_results if m["confidence"] == "high")
            low  = sum(1 for m in qa_results if m["confidence"] == "low")
            unresolved = sum(1 for m in qa_results if not m.get("canonical_formula"))
            print(f"  confidence: {high} high / {low} low")
            print(f"  unresolved canonical_formula: {unresolved}")
            for m in qa_results[:3]:
                print(f"    {m['raw_composition']!r:30s} → canonical={m.get('canonical_formula')!r}  "
                      f"σ_norm={m.get('normalized_conductivity')}  warnings={m['warnings']}")
    else:
        print("\n  (no tables found — skipping steps 3 & 4)")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_tools.py <path_to_paper.md>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
