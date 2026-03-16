"""
Verifier Agent for the PageIndex extraction pipeline.

Multi-agent hub-and-spoke design using Google ADK sub_agents:

  verifier_agent (orchestrator, root)
    ├── log_inspector_agent    — reads extraction_log.jsonl, classifies failures
    ├── failure_recovery_agent — re-runs failed images / tables / text sections
    ├── pdf_verifier_agent     — full-paper PDF sweep for missing / hallucinated data
    └── data_curator_agent     — applies corrections, flags, and direct inserts

The root agent delegates to each specialist via ADK's automatic sub_agent
routing (LLM reads each sub-agent's description and hands off accordingly).

Uses Google ADK (LlmAgent) with gemini-3-flash-preview at temperature=1.0
and the v1alpha API version required by Gemini 3 models.
"""
import os
import re
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

VERIFIER_MODEL = "gemini-3-flash-preview"
_GEN_CONFIG = genai_types.GenerateContentConfig(temperature=1.0)


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})


# =============================================================================
# Tools — unchanged business logic
# =============================================================================

def read_extraction_log(log_path: str) -> Dict[str, Any]:
    """
    Reads the structured extraction log and returns a summary of failures.

    Args:
        log_path: Absolute path to extraction_log.jsonl produced by the pipeline.

    Returns:
        A dict with:
        - 'success_count': number of assets that completed successfully
        - 'failure_count': number of failed assets
        - 'failures': list of full failure event dicts (each has asset_type, asset_id,
          error_type, and 'extra' fields like image_path / section_content / table_content)
        - 'failure_summary': failures grouped by error_type
    """
    path = Path(log_path)
    if not path.exists():
        return {"error": f"Log file not found: {log_path}", "failures": [], "success_count": 0, "failure_count": 0}

    events: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    all_failures = [e for e in events if e.get("status") == "failed"]
    successes = [e for e in events if e.get("status") == "success"]

    # Deduplicate by asset_id — keep only the last failure per asset so the
    # recovery agent calls each re-run tool exactly once.
    seen: Dict[str, Dict] = {}
    for ev in all_failures:
        seen[ev.get("asset_id", "")] = ev
    failures = list(seen.values())

    summary: Dict[str, List] = {}
    for ev in failures:
        etype = ev.get("error_type", "unknown")
        summary.setdefault(etype, []).append({
            "asset_type": ev.get("asset_type"),
            "asset_id": ev.get("asset_id"),
        })

    return {
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
        "failure_summary": summary,
    }


async def rerun_image_extraction(image_path: str, caption: str, results_json_path: str) -> Dict[str, Any]:
    """
    Re-runs extraction on a single failed image and appends any recovered
    measurements to the 'recovered_measurements' key in robust_results_v8.json.

    Args:
        image_path: Absolute path to the image file (png/jpg).
        caption: Figure caption or context string from the log's 'extra' field.
        results_json_path: Absolute path to robust_results_v8.json.

    Returns:
        A dict with 'recovered_count' (int) and 'measurements' (list).
        Returns {'error': ..., 'recovered_count': 0} on failure.
    """
    from extraction_logic import process_image, VISION_MODEL
    from scifigure_parser import SciFigureParser

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    sf_parser = SciFigureParser(api_key=api_key, model_name=VISION_MODEL, debug=False, save_debug=False)

    context_dict = {
        "caption": caption,
        "id": Path(image_path).stem,
        "section_title": "Unknown",
        "section": "",
    }

    try:
        result, _raw, success = await asyncio.wait_for(
            process_image(client, VISION_MODEL, Path(image_path), context_dict, sf_parser=sf_parser),
            timeout=240,
        )
    except Exception as e:
        return {"error": str(e), "recovered_count": 0, "measurements": []}
    finally:
        await client.aio.aclose()

    measurements = getattr(result, "measurements", None) or []
    if not success or not measurements:
        return {"recovered_count": 0, "measurements": []}

    recovered = [m.model_dump() for m in measurements]
    _append_recovered_measurements(results_json_path, recovered,
                                   source=f"verifier_rerun:image:{Path(image_path).name}")
    return {"recovered_count": len(recovered), "measurements": recovered}


async def rerun_table_extraction(
    table_id: str,
    table_caption: str,
    table_content: str,
    results_json_path: str,
) -> Dict[str, Any]:
    """
    Re-runs extraction on a single failed table and appends recovered
    measurements to robust_results_v8.json.

    Args:
        table_id: Table identifier from the log (e.g. "Table 2").
        table_caption: Table caption string from the log's 'extra.table_caption'.
        table_content: Full Markdown table content from the log's 'extra.table_content'.
        results_json_path: Absolute path to robust_results_v8.json.

    Returns:
        A dict with 'recovered_count' and 'measurements'.
    """
    import extraction_logic
    from extraction_logic import process_table_node, TEXT_MODEL

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    extraction_logic.FILE_DIR = str(Path(results_json_path).parent)

    table_data = {"caption": table_caption, "content": table_content, "tab_id": table_id}

    try:
        result, _raw, success = await asyncio.wait_for(
            process_table_node(client, TEXT_MODEL, table_data),
            timeout=240,
        )
    except Exception as e:
        return {"error": str(e), "recovered_count": 0, "measurements": []}
    finally:
        await client.aio.aclose()

    measurements = getattr(result, "measurements", None) or []
    if not success or not measurements:
        return {"recovered_count": 0, "measurements": []}

    recovered = [m.model_dump() for m in measurements]
    _append_recovered_measurements(results_json_path, recovered,
                                   source=f"verifier_rerun:table:{table_id}")
    return {"recovered_count": len(recovered), "measurements": recovered}


async def rerun_text_extraction(
    section_title: str,
    section_content: str,
    results_json_path: str,
) -> Dict[str, Any]:
    """
    Re-runs extraction on a single failed text section and appends recovered
    measurements to robust_results_v8.json.

    Args:
        section_title: Section title from the log's 'extra.section_title'.
        section_content: Full section text from the log's 'extra.section_content'.
        results_json_path: Absolute path to robust_results_v8.json.

    Returns:
        A dict with 'recovered_count' and 'measurements'.
    """
    import extraction_logic
    from extraction_logic import process_text, TEXT_MODEL

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    extraction_logic.FILE_DIR = str(Path(results_json_path).parent)

    try:
        result, _raw, success = await asyncio.wait_for(
            process_text(client, TEXT_MODEL, section_content, section_title),
            timeout=200,
        )
    except Exception as e:
        return {"error": str(e), "recovered_count": 0, "measurements": []}
    finally:
        await client.aio.aclose()

    measurements = getattr(result, "measurements", None) or []
    if not success or not measurements:
        return {"recovered_count": 0, "measurements": []}

    recovered = [m.model_dump() for m in measurements]
    _append_recovered_measurements(results_json_path, recovered,
                                   source=f"verifier_rerun:text:{section_title}")
    return {"recovered_count": len(recovered), "measurements": recovered}


def verify_against_pdf(pdf_path: str, results_json_path: str) -> Dict[str, Any]:
    """
    Full-paper verification pass: uploads the original PDF to gemini-3-flash-preview
    and asks it to identify measurements missing from the extraction results or likely
    hallucinated. Saves a verification_report.json alongside the results file.

    Args:
        pdf_path: Absolute path to the original PDF file.
        results_json_path: Absolute path to robust_results_v8.json.

    Returns:
        A dict with 'missing_candidates', 'hallucination_flags', and
        'verification_notes'. Also writes verification_report.json.
        Returns {'error': ...} if the PDF cannot be found or the call fails.
    """
    if not pdf_path or not Path(pdf_path).exists():
        return {"error": f"PDF not found: {pdf_path}"}

    with open(results_json_path, encoding="utf-8") as f:
        results = json.load(f)
    measurements = results.get("measurements", [])

    meas_summary = json.dumps([
        {
            "composition": m.get("canonical_formula") or m.get("raw_composition"),
            "conductivity_S_cm": m.get("normalized_conductivity"),
            "temperature_C": m.get("normalized_temperature_c"),
            "source": m.get("source"),
        }
        for m in measurements
    ], indent=2)

    prompt = f"""You are a materials science expert reviewing an automated extraction of ionic conductivity data from this scientific paper.

The automated pipeline has already extracted the following {len(measurements)} measurements:
{meas_summary}

Your tasks:
1. MISSING: Identify ionic conductivity measurements clearly reported in the paper (in text, tables, or figures) that are NOT in the extracted list. For each, provide: composition, conductivity value with units, temperature, and the figure/table/section where it appears.
2. HALLUCINATIONS: Flag any extracted measurements that do NOT appear to be supported by the paper (wrong values, compositions absent from the paper, etc.).
3. NOTES: Any other significant data quality observations (unit errors, temperature mismatches, etc.).

Respond as JSON with exactly this structure:
{{
  "missing_candidates": [
    {{"composition": "...", "conductivity": "...", "temperature": "...", "location": "...", "confidence": "high|medium"}}
  ],
  "hallucination_flags": [
    {{"composition": "...", "conductivity": "...", "reason": "..."}}
  ],
  "verification_notes": "..."
}}"""

    client = _get_client()
    try:
        uploaded = client.files.upload(
            file=pdf_path,
            config=genai_types.UploadFileConfig(mime_type="application/pdf"),
        )
        response = client.models.generate_content(
            model=VERIFIER_MODEL,
            contents=[
                genai_types.Part(
                    file_data=genai_types.FileData(
                        mime_type="application/pdf",
                        file_uri=uploaded.uri,
                    ),
                    media_resolution={"level": "media_resolution_medium"},
                ),
                genai_types.Part(text=prompt),
            ],
            config=genai_types.GenerateContentConfig(
                temperature=1.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        return {"error": str(e)}

    try:
        report = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError):
        report = {"raw_response": getattr(response, "text", str(response)),
                  "parse_error": "Could not parse JSON from model response"}

    report_path = Path(results_json_path).parent / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"   Verification report saved to {report_path}")

    return report


def correct_measurement(
    results_json_path: str,
    composition: str,
    wrong_conductivity: float,
    correct_conductivity: float,
    note: str,
) -> Dict[str, Any]:
    """
    Applies a conductivity correction to a specific measurement in robust_results_v8.json.
    Call this when the verification report's hallucination reason contains an explicit
    correct value (e.g. "paper reports 1.31 × 10^-4 S/cm").

    Args:
        results_json_path: Path to robust_results_v8.json.
        composition: Canonical formula or raw composition of the measurement to fix.
        wrong_conductivity: The incorrect normalized_conductivity currently in the file.
        correct_conductivity: The true conductivity value in S/cm as a float.
        note: Short explanation of the correction.

    Returns:
        Dict with 'success' (bool), 'updated_composition' (str), and
        'old_value' / 'new_value' if matched.
    """
    with open(results_json_path, encoding="utf-8") as f:
        data = json.load(f)
    measurements = data.get("measurements", [])

    comp_lower = composition.lower()
    best_idx = None
    best_score = -1

    for i, m in enumerate(measurements):
        stored_comp = (m.get("canonical_formula") or m.get("raw_composition") or "").lower()
        stored_cond = m.get("normalized_conductivity")

        flag_words = set(re.split(r'\W+', comp_lower)) - {'', 'wt', 's', 'cm'}
        stored_words = set(re.split(r'\W+', stored_comp))
        overlap = len(flag_words & stored_words) / max(len(flag_words), 1)

        cond_match = 1.0
        if stored_cond is not None and wrong_conductivity != 0:
            import math
            try:
                log_diff = abs(math.log10(abs(stored_cond)) - math.log10(abs(wrong_conductivity)))
                cond_match = max(0.0, 1.0 - log_diff / 3)
            except Exception:
                cond_match = 0.0

        score = overlap * 0.6 + cond_match * 0.4
        if score > best_score and overlap > 0.4:
            best_score = score
            best_idx = i

    if best_idx is None:
        return {"success": False, "reason": "No matching measurement found"}

    m = measurements[best_idx]
    old_val = m.get("normalized_conductivity")
    m["normalized_conductivity"] = correct_conductivity
    m["confidence"] = "corrected"
    m.setdefault("warnings", []).append(
        f"Verifier correction: {old_val} → {correct_conductivity:.3e} S/cm. {note}"
    )
    m["correction_note"] = note

    data["measurements"] = measurements
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"   correct_measurement: {m.get('canonical_formula')} {old_val} → {correct_conductivity:.3e}")
    return {
        "success": True,
        "updated_composition": m.get("canonical_formula") or m.get("raw_composition"),
        "old_value": old_val,
        "new_value": correct_conductivity,
    }


def add_measurement(
    results_json_path: str,
    composition: str,
    conductivity_s_cm: float,
    temperature_c: float,
    source_location: str,
    note: str = "",
) -> Dict[str, Any]:
    """
    Directly inserts a missing measurement into robust_results_v8.json under
    'recovered_measurements' when the PDF verification report already provides
    sufficient information (composition + conductivity + temperature).

    Args:
        results_json_path: Path to robust_results_v8.json.
        composition: Material composition string as reported by the verifier.
        conductivity_s_cm: Ionic conductivity in S/cm (e.g. 6.17e-6).
        temperature_c: Temperature in Celsius (e.g. 20.0).
        source_location: Where in the paper this came from (e.g. "Table 1, page 4").
        note: Optional extra context from the verification report.

    Returns:
        Dict with 'success' (bool) and the inserted measurement dict.
    """
    measurement = {
        "raw_composition": composition,
        "canonical_formula": composition,
        "normalized_conductivity": conductivity_s_cm,
        "normalized_temperature_c": temperature_c,
        "source": "verifier_added",
        "confidence": "verifier_added",
        "recovery_source": f"verifier_direct:{source_location}",
        "warnings": [
            f"Added directly by verifier from PDF sweep. Source: {source_location}."
            + (f" Note: {note}" if note else "")
        ],
    }
    _append_recovered_measurements(results_json_path, [measurement],
                                   source=f"verifier_direct:{source_location}")
    print(f"   add_measurement: {composition} σ={conductivity_s_cm:.3e} T={temperature_c}°C")
    return {"success": True, "measurement": measurement}


def flag_measurement(
    results_json_path: str,
    composition: str,
    conductivity: float,
    warning: str,
) -> Dict[str, Any]:
    """
    Marks a specific measurement as low-confidence with a hallucination warning.

    Args:
        results_json_path: Path to robust_results_v8.json.
        composition: Canonical formula or raw composition (partial match).
        conductivity: The normalized_conductivity value to help identify the row.
        warning: The hallucination reason to attach as a warning.

    Returns:
        Dict with 'success' (bool) and 'updated_composition' (str).
    """
    with open(results_json_path, encoding="utf-8") as f:
        data = json.load(f)
    measurements = data.get("measurements", [])

    comp_lower = composition.lower()
    best_idx = None
    best_score = -1

    for i, m in enumerate(measurements):
        stored_comp = (m.get("canonical_formula") or m.get("raw_composition") or "").lower()
        stored_cond = m.get("normalized_conductivity")

        flag_words = set(re.split(r'\W+', comp_lower)) - {'', 'wt', 's', 'cm'}
        stored_words = set(re.split(r'\W+', stored_comp))
        overlap = len(flag_words & stored_words) / max(len(flag_words), 1)

        cond_match = 1.0
        if stored_cond is not None and conductivity != 0:
            import math
            try:
                log_diff = abs(math.log10(abs(stored_cond)) - math.log10(abs(conductivity)))
                cond_match = max(0.0, 1.0 - log_diff / 3)
            except Exception:
                cond_match = 0.0

        score = overlap * 0.6 + cond_match * 0.4
        if score > best_score and overlap > 0.4:
            best_score = score
            best_idx = i

    if best_idx is None:
        return {"success": False, "reason": "No matching measurement found"}

    m = measurements[best_idx]
    m["confidence"] = "low"
    m.setdefault("warnings", []).append(f"Hallucination flag: {warning}")
    m["hallucination_warning"] = warning

    data["measurements"] = measurements
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"   flag_measurement: {m.get('canonical_formula')} marked low-confidence")
    return {
        "success": True,
        "updated_composition": m.get("canonical_formula") or m.get("raw_composition"),
    }


# =============================================================================
# Helper
# =============================================================================

def _append_recovered_measurements(results_json_path: str, measurements: List[Dict], source: str) -> None:
    """Appends recovered measurements under 'recovered_measurements' in robust_results_v8.json."""
    path = Path(results_json_path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    existing: List[Dict] = data.get("recovered_measurements", [])
    for m in measurements:
        m["recovery_source"] = source
    existing.extend(measurements)
    data["recovered_measurements"] = existing
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# Specialist Sub-Agents
# =============================================================================

log_inspector_agent = LlmAgent(
    model=VERIFIER_MODEL,
    name="log_inspector_agent",
    description=(
        "Reads and classifies the pipeline extraction log. Use this agent when the task "
        "is to inspect extraction_log.jsonl and understand what failed and why."
    ),
    generate_content_config=_GEN_CONFIG,
    instruction="""You are the Log Inspector for the PageIndex extraction pipeline.

Your sole job: read and classify the extraction log, then return a structured failure report.

STEP 1 — Read the log
Call `read_extraction_log(log_path)` using the log_path value from the user message.

STEP 2 — Classify failures
Group failures by error_type: timeout, exception, model_overloaded.
For each failure, record:
  - asset_type (image / table / text)
  - asset_id
  - error_type
  - the relevant 'extra' fields needed to re-run:
      image  → image_path, caption
      table  → table_id, table_caption, table_content
      text   → section_title, section_content

STEP 3 — Return structured report
Return a JSON-compatible summary with all failure details including the 'extra' fields.
Prioritise timeout failures — they are most likely to succeed on retry.
Do NOT attempt any re-extractions.
""",
    tools=[read_extraction_log],
)


failure_recovery_agent = LlmAgent(
    model=VERIFIER_MODEL,
    name="failure_recovery_agent",
    description=(
        "Re-runs failed image, table, and text extractions from the pipeline. Use this agent "
        "when the task is to retry failed extractions from a failure report."
    ),
    generate_content_config=_GEN_CONFIG,
    instruction="""You are the Failure Recovery specialist for the PageIndex extraction pipeline.

You receive a failure report and a results_json_path. Re-extract every failed asset.

RULES
- Process timeout failures first, then model_overloaded, then exception.
- For each failure call the matching tool:
    asset_type "image" → `rerun_image_extraction(image_path, caption, results_json_path)`
    asset_type "table" → `rerun_table_extraction(table_id, table_caption, table_content, results_json_path)`
    asset_type "text"  → `rerun_text_extraction(section_title, section_content, results_json_path)`
- Use the values from the failure's 'extra' dict as arguments.
- If a re-run returns an error, note it and continue to the next asset.

Return a summary: attempted count, recovered count, total measurements recovered, and
a list of assets that failed again with their errors.
""",
    tools=[rerun_image_extraction, rerun_table_extraction, rerun_text_extraction],
)


pdf_verifier_agent = LlmAgent(
    model=VERIFIER_MODEL,
    name="pdf_verifier_agent",
    description=(
        "Performs a full-paper PDF sweep to find measurements missed by the pipeline or "
        "likely hallucinated. Use this agent when the task is to verify extraction completeness "
        "against the original PDF."
    ),
    generate_content_config=_GEN_CONFIG,
    instruction="""You are the PDF Verifier for the PageIndex extraction pipeline.

Your job: compare the original paper against the extraction results.

STEP 1 — Run the PDF sweep
Call `verify_against_pdf(pdf_path, results_json_path)` using the paths from the user message.

STEP 2 — Annotate the report
For each missing candidate, note:
  - "direct_insert" if composition + conductivity + temperature are ALL present and specific
  - "re_extract" if any field is missing or too vague

For each hallucination flag, note:
  - "correct" if the reason text contains an explicit numeric value (extract it as a float)
  - "flag" if the reason is ambiguous or no correct value can be determined

STEP 3 — Return the full annotated report including all original fields plus your annotations.
""",
    tools=[verify_against_pdf],
)


data_curator_agent = LlmAgent(
    model=VERIFIER_MODEL,
    name="data_curator_agent",
    description=(
        "Applies corrections, flags, and direct inserts based on the PDF verification report. "
        "Use this agent when the task is to act on hallucination flags and missing candidates "
        "from a verification report."
    ),
    generate_content_config=_GEN_CONFIG,
    instruction="""You are the Data Curator for the PageIndex extraction pipeline.

You receive an annotated verification report and a results_json_path. Apply all fixes.

PART A — Handle hallucination_flags
For each entry:
  - action "correct": call `correct_measurement(results_json_path, composition,
      wrong_conductivity, correct_conductivity, reason)`
    Parse the correct value from the reason text as a float in S/cm.
  - action "flag" or no clear correct value: call `flag_measurement(results_json_path,
      composition, conductivity, reason)`

PART B — Recover missing_candidates
For each entry:
  - recovery_strategy "direct_insert": call `add_measurement(results_json_path,
      composition, conductivity_s_cm, temperature_c, location, note)`
    Parse conductivity to a float in S/cm (e.g. "1.2×10⁻⁴ S/cm" → 1.2e-4).
    Parse temperature to a float in °C (e.g. "25°C" → 25.0).
  - recovery_strategy "re_extract": note as "needs_reextraction" — do not attempt.

Return a curation summary: hallucinations_corrected, hallucinations_flagged,
missing_direct_inserted, missing_needs_reextraction list, and any errors.
""",
    tools=[correct_measurement, add_measurement, flag_measurement],
)


# =============================================================================
# Runner — explicit sequential orchestration
#
# Rather than relying on a 5th LLM agent to route between the 4 specialists
# (which causes ADK to terminate the run_async stream after the first
# sub-agent final response), we drive the sequence from Python directly.
# Each specialist still runs as a full ADK LlmAgent with its own tools and
# reasoning; Python just decides what order to call them in.
# =============================================================================

async def _run_agent(agent: LlmAgent, prompt: str, step_label: str) -> str:
    """
    Run a single ADK LlmAgent to completion and return its final text response.
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="pageindex_verifier",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="pageindex_verifier", user_id="system"
    )

    final = ""
    async for event in runner.run_async(
        user_id="system",
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        ),
    ):
        tool_calls = event.get_function_calls()
        tool_responses = event.get_function_responses()
        if tool_calls:
            for tc in tool_calls:
                print(f"  [{step_label} TOOL ] {tc.name}({str(tc.args)[:120]})")
        elif tool_responses:
            for tr in tool_responses:
                print(f"  [{step_label} RESP ] {tr.name}: {str(tr.response)[:120]}")

        if event.is_final_response() and event.content and event.content.parts:
            text_parts = [p.text for p in event.content.parts if p.text]
            if text_parts:
                final = "\n".join(text_parts)
                print(f"  [{step_label} DONE ] {final[:160]}")

    return final


async def run_verifier(log_path: str, results_json_path: str, pdf_path: str) -> str:
    """
    Run the 4-stage verification pipeline after the main extraction completes.

    Stages are sequenced explicitly from Python:
      1. log_inspector_agent   — read & classify failures
      2. failure_recovery_agent — retry timed-out / errored assets
      3. pdf_verifier_agent    — full-paper PDF sweep (skipped if no PDF)
      4. data_curator_agent    — apply corrections & inserts (skipped if no PDF)

    Args:
        log_path: Path to extraction_log.jsonl
        results_json_path: Path to robust_results_v8.json
        pdf_path: Path to the original PDF file (empty string to skip PDF steps)

    Returns:
        A JSON string summarising each stage's outcome.
    """
    summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage 1 — Log Inspection
    # ------------------------------------------------------------------
    print("\n--- [Verifier] Stage 1: Log Inspection ---")
    log_report = await _run_agent(
        log_inspector_agent,
        f"Inspect the extraction log and return a detailed structured failure report "
        f"including all 'extra' fields (image_path, caption, table_content, etc.) "
        f"needed to re-run each failed asset.\nlog_path: {log_path}",
        step_label="LOG",
    )
    summary["log_inspection"] = log_report

    # ------------------------------------------------------------------
    # Stage 2 — Failure Recovery
    # ------------------------------------------------------------------
    print("\n--- [Verifier] Stage 2: Failure Recovery ---")
    recovery_report = await _run_agent(
        failure_recovery_agent,
        f"Re-run all failed extractions from the failure report below.\n"
        f"results_json_path: {results_json_path}\n\n"
        f"Failure Report:\n{log_report}",
        step_label="RECOVERY",
    )
    summary["failure_recovery"] = recovery_report

    # ------------------------------------------------------------------
    # Stage 3 — PDF Verification (skip if no PDF supplied)
    # ------------------------------------------------------------------
    pdf_report = ""
    if pdf_path and Path(pdf_path).exists():
        print("\n--- [Verifier] Stage 3: PDF Verification ---")
        pdf_report = await _run_agent(
            pdf_verifier_agent,
            f"Verify the extraction results against the original PDF.\n"
            f"pdf_path: {pdf_path}\n"
            f"results_json_path: {results_json_path}",
            step_label="PDF",
        )
        summary["pdf_verification"] = pdf_report
    else:
        print("\n--- [Verifier] Stage 3: PDF Verification SKIPPED (no PDF found) ---")
        summary["pdf_verification"] = "skipped — no PDF supplied"

    # ------------------------------------------------------------------
    # Stage 4 — Data Curation (only if PDF verification produced a report)
    # ------------------------------------------------------------------
    if pdf_report:
        print("\n--- [Verifier] Stage 4: Data Curation ---")
        curation_report = await _run_agent(
            data_curator_agent,
            f"Apply all corrections and inserts from the verification report below.\n"
            f"results_json_path: {results_json_path}\n\n"
            f"Verification Report:\n{pdf_report}",
            step_label="CURATE",
        )
        summary["data_curation"] = curation_report
    else:
        print("\n--- [Verifier] Stage 4: Data Curation SKIPPED (no PDF report) ---")
        summary["data_curation"] = "skipped — no PDF verification report"

    return json.dumps(summary, indent=2)
