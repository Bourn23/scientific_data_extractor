"""
PageIndex: Simple API for extracting structured ionic-conductivity
measurements from scientific papers.

Usage
-----
    from parse import extract

    # Extract from a single paper (markdown + assets in same folder)
    result = extract("path/to/paper.md")
    print(result.measurements)        # list of MeasuredPoint dicts
    print(result.summary)             # quick stats
    print(result.cost)                # API cost breakdown

    # Extract with PDF verification
    result = extract("path/to/paper.md", pdf="path/to/paper.pdf", verify=True)
    print(result.verification_report)

    # Extract from multiple papers
    results = extract(["paper1.md", "paper2.md"])
    for r in results:
        print(f"{r.doc_name}: {r.n_measurements} measurements, ${r.cost['total_usd']:.4f}")

    # Inspect the prompts and schemas used
    from parse import show_schema, show_prompts
    show_schema()   # prints the Pydantic MeasuredPoint model
    show_prompts()  # prints all extraction prompts
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Container for extraction results from a single paper."""

    doc_name: str
    measurements: List[dict]
    material_definitions: List[dict]
    paper_context: dict
    stats: dict
    cost: dict
    config: dict
    is_review_article: bool
    execution_time_seconds: float
    verification_report: Optional[str] = None

    # --- Convenience properties -------------------------------------------

    @property
    def n_measurements(self) -> int:
        return len(self.measurements)

    @property
    def summary(self) -> str:
        lines = [
            f"Document : {self.doc_name}",
            f"Measurements : {self.n_measurements}",
            f"Sources : {self._source_breakdown()}",
            f"Execution time : {self.execution_time_seconds:.1f}s",
            f"API cost : ${self.cost.get('total_usd', 0):.4f}",
        ]
        if self.is_review_article:
            lines.insert(1, "Type : Review article")
        return "\n".join(lines)

    @property
    def markdown(self) -> str:
        """Render measurements as a Markdown table."""
        if not self.measurements:
            return "_No measurements extracted._"
        cols = [
            ("Composition", "canonical_formula", 40),
            ("Class", "material_class", 10),
            ("σ (S/cm)", "normalized_conductivity", 12),
            ("T (°C)", "normalized_temperature_c", 8),
            ("Source", "source", 8),
            ("Confidence", "confidence", 10),
        ]
        header = "| " + " | ".join(c[0].ljust(c[2]) for c in cols) + " |"
        sep = "| " + " | ".join("-" * c[2] for c in cols) + " |"
        rows = [header, sep]
        for m in self.measurements:
            vals = []
            for _, key, width in cols:
                v = m.get(key)
                if v is None:
                    v = "—"
                elif isinstance(v, float):
                    v = f"{v:.2e}" if abs(v) < 0.01 or abs(v) > 1000 else f"{v:.4f}"
                vals.append(str(v)[:width].ljust(width))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    @property
    def dataframe(self):
        """Return a pandas DataFrame (requires pandas)."""
        import pandas as pd
        return pd.DataFrame(self.measurements)

    def save(self, path: str) -> str:
        """Save results to JSON."""
        out = {
            "doc_name": self.doc_name,
            "extraction_version": "v8",
            "is_review_article": self.is_review_article,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": self.execution_time_seconds,
            "config": self.config,
            "cost_summary": self.cost,
            "stats": self.stats,
            "paper_context": self.paper_context,
            "material_count": self.n_measurements,
            "measurements": self.measurements,
            "material_definitions": self.material_definitions,
        }
        if self.verification_report:
            out["verification_report"] = self.verification_report
        p = Path(path)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return str(p.resolve())

    # --- internals --------------------------------------------------------

    def _source_breakdown(self) -> str:
        from collections import Counter
        counts = Counter(m.get("source", "unknown") for m in self.measurements)
        return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Post-processing helpers (from agent.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def extract(
    source: Union[str, List[str]],
    *,
    pdf: Optional[str] = None,
    verify: bool = False,
    model: str = "gemini-3-flash-preview",
    result_save_dir: Optional[str] = None,
    progress: bool = True,
) -> Union[ExtractionResult, List[ExtractionResult]]:
    """
    Extract structured ionic-conductivity measurements from one or more papers.

    Parameters
    ----------
    source : str or list of str
        Path(s) to markdown file(s) produced by document ingestion.
        Each .md file should sit alongside its extracted assets (images, etc.).
    pdf : str, optional
        Path to the original PDF. Used by the verifier agent for a full-paper
        sweep. If omitted, the pipeline looks for a .pdf in the same directory.
    verify : bool, default False
        If True, run the agentic verifier after extraction to recover failed
        extractions and check for hallucinations.
    model : str, default "gemini-3-flash-preview"
        Gemini model to use for extraction.
    result_save_dir : str, optional
        If provided, save each result JSON to this directory.
    progress : bool, default True
        Print progress messages to stdout.

    Returns
    -------
    ExtractionResult or list of ExtractionResult
    """
    if isinstance(source, (list, tuple)):
        results = []
        for s in source:
            results.append(_extract_single(s, pdf=pdf, verify=verify, model=model,
                                           result_save_dir=result_save_dir, progress=progress))
        return results
    return _extract_single(source, pdf=pdf, verify=verify, model=model,
                           result_save_dir=result_save_dir, progress=progress)


def _extract_single(
    source: str,
    *,
    pdf: Optional[str],
    verify: bool,
    model: str,
    result_save_dir: Optional[str],
    progress: bool,
) -> ExtractionResult:
    """Run extraction on a single markdown file."""
    md_path = Path(source).resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    asset_dir = md_path.parent

    if progress:
        print(f"\n{'=' * 60}")
        print(f"PageIndex — Extracting: {md_path.name}")
        print(f"{'=' * 60}\n")

    # ---- Phase 1: Extraction ----
    from extraction_logic import run_pipeline, VISION_MODEL, TEXT_MODEL, tracker

    # Reset the cost tracker for this run
    tracker.__init__()

    t0 = time.time()
    measurements, material_defs, stats, is_review_article, paper_context = asyncio.run(
        run_pipeline(md_path, asset_dir, model=model)
    )

    assumed = _apply_missing_temperature_assumptions(measurements)
    if assumed and progress:
        print(f"   Applied RT assumption to {assumed} measurement(s).")

    elapsed = round(time.time() - t0, 2)

    cost_info = {
        "total_input_tokens": tracker.total_input_tokens,
        "total_output_tokens": tracker.total_output_tokens,
        "total_usd": round(tracker.total_cost_usd, 4),
        "call_counts": dict(tracker.call_counts),
    }

    config_info = {
        "vision_model": VISION_MODEL,
        "text_model": TEXT_MODEL,
        "extraction_model": model,
        "normalization_engine": "v2_hybrid_llm_python",
    }

    if progress:
        tracker.print_summary()

    result = ExtractionResult(
        doc_name=md_path.stem,
        measurements=[m.model_dump() for m in measurements],
        material_definitions=material_defs,
        paper_context=paper_context or {},
        stats=stats,
        cost=cost_info,
        config=config_info,
        is_review_article=is_review_article,
        execution_time_seconds=elapsed,
    )

    # ---- Phase 2: Verification (optional) ----
    if verify:
        result = _run_verification(result, md_path, pdf, progress)

    # ---- Save if requested ----
    if result_save_dir:
        save_dir = Path(result_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"{md_path.stem}_{ts}.json"
        result.save(str(save_path))
        if progress:
            print(f"Results saved to {save_path}")

    return result


def _run_verification(
    result: ExtractionResult,
    md_path: Path,
    pdf: Optional[str],
    progress: bool,
) -> ExtractionResult:
    """Run the verifier agent on extraction results."""
    from verifier_agent import run_verifier

    log_path = md_path.parent / "extraction_log.jsonl"
    out_path = md_path.parent / "robust_results_v8.json"

    # Save intermediate results for verifier
    result.save(str(out_path))

    if not log_path.exists():
        if progress:
            print("No extraction log found — skipping verification.")
        return result

    # Resolve PDF
    if not pdf:
        candidates = list(md_path.parent.glob("*.pdf"))
        pdf = str(candidates[0]) if candidates else ""

    if progress:
        print(f"\n{'=' * 60}")
        print("Running verifier agent...")
        print(f"{'=' * 60}")

    report = asyncio.run(run_verifier(
        log_path=str(log_path),
        results_json_path=str(out_path),
        pdf_path=pdf or "",
    ))

    result.verification_report = report

    # Reload measurements if verifier added recovered ones
    if out_path.exists():
        updated = json.loads(out_path.read_text(encoding="utf-8"))
        if "recovered_measurements" in updated:
            result.measurements.extend(updated["recovered_measurements"])

    if progress:
        print(f"\n{'=' * 60}")
        print("VERIFIER REPORT")
        print(f"{'=' * 60}")
        print(report)

    return result


# ---------------------------------------------------------------------------
# Introspection: show_schema / show_prompts
# ---------------------------------------------------------------------------

def show_schema(as_json: bool = False) -> Optional[str]:
    """
    Display the Pydantic MeasuredPoint schema used for extraction.

    Parameters
    ----------
    as_json : bool
        If True, return JSON schema string instead of printing.
    """
    from extraction_logic import MeasuredPoint
    schema = MeasuredPoint.model_json_schema()

    if as_json:
        return json.dumps(schema, indent=2)

    print("=" * 60)
    print("MeasuredPoint — Extraction Schema")
    print("=" * 60)
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, info in props.items():
        req = "*" if name in required else " "
        typ = info.get("type", info.get("anyOf", ""))
        if isinstance(typ, list):
            typ = " | ".join(t.get("type", str(t)) for t in typ)
        elif isinstance(typ, dict):
            typ = str(typ)
        desc = info.get("description", "")
        # Truncate long descriptions for display
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"  {req} {name:<35} {str(typ):<15} {desc}")
    print()
    print(f"Total fields: {len(props)}  |  Required: {len(required)}")
    return None


def show_prompts() -> dict:
    """
    Return the key extraction prompts used by the pipeline.

    Returns a dict with keys: 'text_extraction', 'table_extraction',
    'abbreviation_extraction', 'figure_analysis', 'paper_context'.
    Each value is the prompt template string.
    """
    import inspect
    from extraction_logic import (
        process_text,
        process_table_node,
        extract_abbreviation_map,
        extract_paper_context,
    )

    prompts = {}

    # Extract prompts from function source code
    for name, func in [
        ("text_extraction", process_text),
        ("table_extraction", process_table_node),
        ("abbreviation_extraction", extract_abbreviation_map),
        ("paper_context", extract_paper_context),
    ]:
        source = inspect.getsource(func)
        prompts[name] = {
            "function": func.__name__,
            "source_lines": len(source.splitlines()),
            "source": source,
        }

    return prompts


def get_schema_model():
    """Return the MeasuredPoint Pydantic model class for programmatic use."""
    from extraction_logic import MeasuredPoint
    return MeasuredPoint
