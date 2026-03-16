import os
import re
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

from extraction_logic import (
    MarkdownContextParser,
    MeasurementProcessor,
    validate_formula_stoichiometry,
    normalize_formula_to_reduced,
    _resolve_nasicon_acronym,
    _lookup_abbreviation_map,
    extract_abbreviation_map,
    process_table_node,
    TEXT_MODEL,
    SectionInfo,
)
from scifigure_parser import SciFigureParser

load_dotenv()

VISION_MODEL = "gemini-3-flash-preview"

# =============================================================================
# Module-level client (shared across tools)
# =============================================================================

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# =============================================================================
# 1. Structural & Context Tools (Used by Lead Researcher & Context Librarian)
# =============================================================================

def parse_markdown_structure(file_content: str, asset_dir: str = "") -> Dict[str, Any]:
    """
    Parses the raw Markdown text of a scientific paper to map its structure.

    Args:
        file_content: The raw text of the markdown file.
        asset_dir: (Optional) Absolute path to the directory containing the paper's
            image files. When provided, image entries will include a 'full_path' field
            that can be passed directly to extract_from_plot.

    Returns:
        A dictionary containing:
        - 'title': Document title.
        - 'sections': List of text sections with headers and line numbers.
        - 'images': List of identified image filenames, captions, and full_path.
        - 'tables': List of identified markdown tables.
    """
    from pathlib import Path

    parser = MarkdownContextParser()
    doc_title, sections = parser.parse_structure(file_content)
    images = parser.parse_images(file_content)
    tables = parser.parse_tables(file_content)

    base = Path(asset_dir) if asset_dir else None

    return {
        "title": doc_title,
        "sections": [
            {
                "id": s.id,
                "title": s.title,
                "content_preview": s.content[:800],
                "line_num": s.line_num,
                "end_line_num": s.end_line_num,
            }
            for s in sections if s
        ],
        "images": [
            {
                "filename": img.filename,
                "figure_id": img.id,
                "caption": img.caption,
                "full_path": str(base / img.filename) if base else img.filename,
            }
            for img in images
        ],
        "tables": [
            {
                "tab_id": t.id,
                "caption": t.caption,
                "content": t.content,
            }
            for t in tables
        ],
    }


async def retrieve_document_abbreviations(text_content: str) -> Dict[str, str]:
    """
    Scans the introduction and methods sections to find material abbreviations.
    Calls the LLM to extract abbreviation → canonical formula mappings.

    Args:
        text_content: The combined text of the relevant sections (intro + methods).

    Returns:
        A dictionary mapping abbreviations to canonical formulas
        (e.g., {"LATP03": "Li1.3Al0.3Ti1.7(PO4)3"}).
        May also contain "__material_classes__" metadata key.
    """
    client = _get_client()
    parser = MarkdownContextParser()
    doc_title, sections = parser.parse_structure(text_content)

    abbreviation_map = await extract_abbreviation_map(
        client=client,
        doc_title=doc_title,
        sections=sections,
        model_name=TEXT_MODEL,
    )
    # Remove metadata keys before returning to the agent
    clean_map = {k: v for k, v in abbreviation_map.items() if not k.startswith("__")}
    return clean_map


# =============================================================================
# 2. Extraction Tools (Used by the Data Extractor Agent)
# =============================================================================

class ExtractionResult(BaseModel):
    raw_composition: str
    raw_conductivity: str
    raw_conductivity_unit: str
    raw_temperature: str
    raw_temperature_unit: str
    normalized_conductivity: Optional[float]
    normalized_temperature_c: Optional[float]
    source_type: str  # 'figure', 'table', 'text'
    confidence: str   # 'high', 'low'
    warnings: List[str] = Field(default_factory=list)


async def extract_from_plot(image_path: str, caption: str, x_axis_hint: str = None) -> List[Dict]:
    """
    Extracts ionic conductivity data points from a scientific plot using SciFigureParser.
    Uses a two-step pipeline: subplot detection → crop → data extraction → normalization.

    Args:
        image_path: Absolute path to the figure image file (PNG, JPG, etc.).
        caption: The figure caption or surrounding context text.
        x_axis_hint: (Optional) Expected x-axis type: 'temperature_inverse',
                     'temperature_absolute', or 'stoichiometry'. Passed as a hint.

    Returns:
        A list of measurement dicts containing raw and normalized conductivity/temperature.
        Returns an empty list if no conductivity data is found in the figure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    sf = SciFigureParser(api_key=api_key, model_name=VISION_MODEL, debug=False, save_debug=False)
    processor = MeasurementProcessor()

    # Step 1: Detect subplots
    query = "ionic conductivity vs temperature or stoichiometry"
    figure_analysis = await sf.detect_subplot_async(image_path, query)

    if not figure_analysis:
        return []

    results = []
    subplots = figure_analysis.get("subplots", [])

    for subplot in subplots:
        if not subplot.get("contains_conductivity_data", False):
            continue

        # Step 2: Crop the subplot from the full figure
        box_2d = subplot.get("box_2d", [])
        if len(box_2d) != 4:
            continue

        # box_2d is [ymin, xmin, ymax, xmax] — convert to dict for crop_image
        box_dict = {
            "ymin": box_2d[0],
            "xmin": box_2d[1],
            "ymax": box_2d[2],
            "xmax": box_2d[3],
        }

        # Save cropped image to a temp file so extract_data_async can read it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cropped_path = tmp.name
        sf.crop_image(image_path, box_dict, output_path=cropped_path)

        # Step 3: Build axis hints from detection result for the extraction call
        axis_hints = {
            "x_axis": subplot.get("x_axis", {}),
            "left_y_axis": subplot.get("left_y_axis", {}),
        }
        if subplot.get("right_y_axis"):
            axis_hints["right_y_axis"] = subplot["right_y_axis"]

        # Inject x_axis_hint if provided and detection was uncertain
        if x_axis_hint and axis_hints["x_axis"].get("quantity_type", "other") == "other":
            axis_hints["x_axis"]["quantity_type"] = x_axis_hint

        # Step 4: Extract data points from the cropped subplot
        extraction = await sf.extract_data_async(
            image_path=cropped_path,
            context=caption,
            axis_hints=axis_hints,
        )

        if not extraction:
            continue

        # Step 5: Normalize via MeasurementProcessor
        fig_id = subplot.get("label", "subplot")
        measured_points = processor.process_extraction(extraction, fig_id, caption, axis_hints)

        for m in measured_points:
            results.append({
                "raw_composition": m.raw_composition,
                "raw_conductivity": m.raw_conductivity,
                "raw_conductivity_unit": m.raw_conductivity_unit,
                "raw_temperature": m.raw_temperature,
                "raw_temperature_unit": m.raw_temperature_unit,
                "normalized_conductivity": m.normalized_conductivity,
                "normalized_temperature_c": m.normalized_temperature_c,
                "source_type": "figure",
                "confidence": m.confidence,
                "warnings": m.warnings,
            })

        # Clean up temp file
        try:
            os.unlink(cropped_path)
        except OSError:
            pass

    return results


async def extract_from_markdown_table(table_markdown: str, caption: str) -> List[Dict]:
    """
    Extracts ionic conductivity measurements from a Markdown formatted table.
    Prioritizes room-temperature conductivity (σRT) over activation energy (Ea).
    Handles multi-level headers, NASICON acronyms, and bulk/grain-boundary splits.

    Args:
        table_markdown: The raw markdown string of the table (pipe-delimited).
        caption: The table's caption (e.g., "Table 2. Ionic conductivities at 25°C").

    Returns:
        A list of raw measurement dicts. Values are not yet canonicalized.
    """
    client = _get_client()
    table_data = {
        "content": table_markdown,
        "caption": caption,
        "tab_id": "Table (agent)",
    }

    extraction_result = await process_table_node(
        client=client,
        model=TEXT_MODEL,
        table_data=table_data,
    )

    if not extraction_result:
        return []

    results = []
    for m in extraction_result:
        results.append({
            "raw_composition": m.raw_composition,
            "raw_conductivity": m.raw_conductivity,
            "raw_conductivity_unit": m.raw_conductivity_unit,
            "raw_temperature": m.raw_temperature,
            "raw_temperature_unit": m.raw_temperature_unit,
            "normalized_conductivity": m.normalized_conductivity,
            "normalized_temperature_c": m.normalized_temperature_c,
            "material_class": m.material_class,
            "source_type": "table",
            "confidence": m.confidence,
            "warnings": m.warnings,
        })

    return results


# =============================================================================
# 3. Validation & Physics Tools (Used by the Materials Chemist Agent)
# =============================================================================
# NOTE: The three granular tools below (validate_stoichiometry, resolve_nasicon_or_series,
# normalize_and_check_bounds) are kept as standalone utilities for direct use or testing.
# For agent use, prefer QA_process_measurement_batch which calls all three in a Python
# loop — one tool call instead of 3×N calls for N measurements.

def validate_stoichiometry(chemical_formula: str) -> Dict[str, Any]:
    """
    Validates if a chemical formula has physically plausible stoichiometry for solid-state electrolytes.
    Checks total atom count (3–200), Li range (0.1–50), and O range (1–150).

    Args:
        chemical_formula: The string formula to check (e.g., "Li10GeP2S12", "Li1.3Al0.3Ti1.7(PO4)3").

    Returns:
        A dictionary with:
        - 'is_valid' (bool): Whether the formula passes plausibility checks.
        - 'reduced_formula' (str): GCD-reduced alphabetically sorted formula (if valid).
        - 'error_reason' (str, optional): Explanation if invalid.
    """
    is_valid = validate_formula_stoichiometry(chemical_formula)

    if not is_valid:
        return {
            "is_valid": False,
            "reduced_formula": None,
            "error_reason": (
                f"Formula '{chemical_formula}' failed stoichiometry plausibility check. "
                "Possible causes: total atom count outside [3, 200], Li outside [0.1, 50], "
                "O outside [1, 150], or other element coefficient outside [0.05, 30]."
            ),
        }

    reduced = normalize_formula_to_reduced(chemical_formula)
    return {
        "is_valid": True,
        "reduced_formula": reduced,
        "error_reason": None,
    }


def resolve_nasicon_or_series(raw_name: str, abbreviation_map: Dict[str, str]) -> str:
    """
    Attempts to deterministically resolve acronyms (like LATP03) or series variables
    (x=0.1) into full canonical chemical formulas, without calling an LLM.

    Resolution order:
    1. Paper-specific abbreviation map lookup (most reliable, accounts for paper conventions)
    2. NASICON pattern resolver for LATP/LCTP/LFTP/LTP acronyms

    Args:
        raw_name: The raw extracted name from the figure or table (e.g., "LATP03", "x=0.25").
        abbreviation_map: The dict returned by retrieve_document_abbreviations for this paper.

    Returns:
        The resolved canonical formula string, or None if it cannot be resolved deterministically.
        A grain_boundary or bulk measurement_type suffix is stripped and returned in the result.
    """
    # Pass 1: Paper-specific abbreviation map
    canonical, measurement_type = _lookup_abbreviation_map(raw_name, abbreviation_map)
    if canonical:
        result = canonical
        if measurement_type:
            result += f" [{measurement_type}]"
        return result

    # Pass 2: Deterministic NASICON resolver
    formula, measurement_type = _resolve_nasicon_acronym(raw_name)
    if formula:
        result = formula
        if measurement_type:
            result += f" [{measurement_type}]"
        return result

    return None


def normalize_and_check_bounds(
    raw_cond: float, cond_unit: str,
    raw_temp: float, temp_unit: str,
    is_arrhenius: bool = False
) -> Dict[str, Any]:
    """
    Converts raw temperatures to Celsius and conductivity to S/cm.
    Applies physical plausibility bounds for solid-state electrolyte materials.

    Args:
        raw_cond: The extracted conductivity value as a float.
        cond_unit: Unit string (e.g., 'mS/cm', 'log(S/cm)', 'S/m', 'μS/cm').
        raw_temp: The extracted temperature value as a float.
        temp_unit: Unit string (e.g., '1000/T', 'K', '°C', 'Celsius').
        is_arrhenius: True if the source plot had 1000/T on x-axis (negative slope expected).

    Returns:
        Dictionary containing:
        - 'normalized_cond_s_cm' (float): Conductivity in S/cm (None if σ₀ / pre-exponential).
        - 'normalized_temp_c' (float): Temperature in Celsius.
        - 'warnings' (List[str]): Physical plausibility warnings.
    """
    processor = MeasurementProcessor()
    warnings = []

    # Build synthetic axis definition dicts that MeasurementProcessor's methods expect
    if is_arrhenius or "1000/t" in temp_unit.lower() or "1000" in temp_unit:
        x_quantity = "temperature_inverse"
    elif temp_unit.lower() in ("k", "kelvin") or (
        "k" in temp_unit.lower() and "c" not in temp_unit.lower()
    ):
        x_quantity = "temperature_absolute"
    else:
        x_quantity = "temperature_absolute"

    unit_lower = temp_unit.lower()
    temp_is_kelvin = "k" in unit_lower and "c" not in unit_lower

    x_axis_def = {
        "quantity_type": x_quantity,
        "unit": temp_unit,
        "scale_type": "reciprocal" if x_quantity == "temperature_inverse" else "linear",
    }
    y_axis_def = {
        "unit": cond_unit,
        "title_text": "",
        "scale_type": "log" if "log" in cond_unit.lower() else "linear",
    }

    # Normalize temperature
    temp_c = processor._normalize_temperature(raw_temp, x_axis_def)

    # If absolute temperature and unit doesn't clarify K vs C, apply heuristic
    if x_quantity == "temperature_absolute" and temp_is_kelvin:
        temp_c = raw_temp - 273.15
    elif x_quantity == "temperature_absolute" and not temp_is_kelvin:
        temp_c = raw_temp  # Assume Celsius

    # Normalize conductivity
    temperature_k = (temp_c + 273.15) if temp_c is not None else None
    cond_s_cm = processor._normalize_conductivity(raw_cond, y_axis_def, temperature_k)

    # Physical bounds check
    if cond_s_cm is not None:
        if cond_s_cm > processor.MAX_REALISTIC_COND_RT:
            warnings.append(
                f"Conductivity {cond_s_cm:.2e} S/cm exceeds realistic maximum "
                f"({processor.MAX_REALISTIC_COND_RT} S/cm). Likely an extraction artifact."
            )
        if cond_s_cm < processor.MIN_REALISTIC_COND:
            warnings.append(
                f"Conductivity {cond_s_cm:.2e} S/cm is below minimum realistic threshold "
                f"({processor.MIN_REALISTIC_COND} S/cm). Possibly noise or a unit error."
            )
        if temp_c is not None and temp_c < -100:
            warnings.append(f"Temperature {temp_c:.1f}°C is extremely low. Check unit conversion.")
        if temp_c is not None and temp_c > 1000:
            warnings.append(f"Temperature {temp_c:.1f}°C is extremely high for a solid electrolyte.")

    return {
        "normalized_cond_s_cm": cond_s_cm,
        "normalized_temp_c": temp_c,
        "warnings": warnings,
    }


# =============================================================================
# 4. Batch QA Tool (Primary tool for the Materials Chemist Agent)
# =============================================================================

def QA_process_measurement_batch(
    raw_measurements: List[Dict],
    abbreviation_map: Dict[str, str],
) -> List[Dict]:
    """
    Runs the full QA pipeline on every measurement in one Python loop.
    Replaces what would otherwise be 3×N sequential LLM tool calls with a single call.

    For each measurement, in order:
      1. Resolve the composition name (abbreviation map → NASICON resolver).
      2. Validate stoichiometry of the resolved formula.
      3. Check that conductivity and temperature are within physical bounds.
      4. Set confidence to "low" if any warnings were raised; preserve "high" otherwise.

    Args:
        raw_measurements: List of measurement dicts as returned by extract_from_plot or
            extract_from_markdown_table. Each dict must contain at minimum:
            - raw_composition (str)
            - raw_conductivity (str or float)
            - raw_conductivity_unit (str)
            - raw_temperature (str or float)
            - raw_temperature_unit (str)
            Other keys (source_type, material_class, etc.) are passed through unchanged.
        abbreviation_map: Dict returned by retrieve_document_abbreviations for this paper.

    Returns:
        The same list of measurement dicts, each augmented with:
        - 'canonical_formula' (str | None): Resolved formula, or None if unresolvable.
        - 'reduced_formula' (str | None): GCD-reduced form for cross-paper matching.
        - 'stoichiometry_valid' (bool): Whether the formula passed plausibility checks.
        - 'normalized_conductivity' (float | None): In S/cm (overwritten from extraction).
        - 'normalized_temperature_c' (float | None): In Celsius (overwritten from extraction).
        - 'confidence' (str): 'high' or 'low'.
        - 'warnings' (List[str]): All accumulated warnings from all three QA steps.
    """
    processed = []

    for m in raw_measurements:
        result = dict(m)  # shallow copy — preserve source_type, material_class, etc.
        all_warnings = list(m.get("warnings", []))

        # ------------------------------------------------------------------
        # Step 1: Name resolution
        # ------------------------------------------------------------------
        raw_comp = m.get("raw_composition", "")
        canonical = resolve_nasicon_or_series(raw_comp, abbreviation_map)

        # Strip the measurement_type suffix that resolve_ may append (e.g., " [bulk]")
        measurement_type_tag = None
        if canonical and " [" in canonical:
            canonical, tag = canonical.rsplit(" [", 1)
            measurement_type_tag = tag.rstrip("]")

        result["canonical_formula"] = canonical
        if measurement_type_tag:
            result["measurement_condition"] = measurement_type_tag

        # ------------------------------------------------------------------
        # Step 2: Stoichiometry validation
        # ------------------------------------------------------------------
        formula_to_check = canonical or raw_comp
        stoich = validate_stoichiometry(formula_to_check)
        result["stoichiometry_valid"] = stoich["is_valid"]
        result["reduced_formula"] = stoich.get("reduced_formula")
        if not stoich["is_valid"] and stoich.get("error_reason"):
            all_warnings.append(stoich["error_reason"])

        # ------------------------------------------------------------------
        # Step 3: Physical bounds check
        # ------------------------------------------------------------------
        try:
            raw_cond = float(m.get("raw_conductivity", 0))
        except (TypeError, ValueError):
            raw_cond = 0.0
            all_warnings.append(f"Could not parse raw_conductivity: {m.get('raw_conductivity')!r}")

        try:
            raw_temp = float(m.get("raw_temperature", 25))
        except (TypeError, ValueError):
            raw_temp = 25.0
            all_warnings.append(f"Could not parse raw_temperature: {m.get('raw_temperature')!r}")

        cond_unit = m.get("raw_conductivity_unit", "S/cm")
        temp_unit = m.get("raw_temperature_unit", "°C")

        norm = normalize_and_check_bounds(raw_cond, cond_unit, raw_temp, temp_unit)
        result["normalized_conductivity"] = norm["normalized_cond_s_cm"]
        result["normalized_temperature_c"] = norm["normalized_temp_c"]
        all_warnings.extend(norm["warnings"])

        # ------------------------------------------------------------------
        # Step 4: Final confidence roll-up
        # ------------------------------------------------------------------
        # Preserve "high" only if no new warnings were added and the original was "high"
        original_confidence = m.get("confidence", "high")
        new_warnings = all_warnings[len(m.get("warnings", [])):]  # warnings added by QA
        if new_warnings or original_confidence == "low":
            result["confidence"] = "low"
        else:
            result["confidence"] = "high"

        result["warnings"] = all_warnings
        processed.append(result)

    return processed
