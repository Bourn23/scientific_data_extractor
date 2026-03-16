import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Import the tools defined in tools.py
from tools import (
    parse_markdown_structure,
    retrieve_document_abbreviations,
    extract_from_plot,
    extract_from_markdown_table,
    QA_process_measurement_batch,
)

load_dotenv()

APP_NAME = "pageindex_extractor"

# =============================================================================
# 1. The Context Librarian
# =============================================================================
librarian_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="context_librarian",
    description=(
        "Extracts global paper context. Route to this agent to build the abbreviation map "
        "from a paper's introduction or methods sections. Returns a dict mapping "
        "sample labels like 'LATP03' to canonical formulas like 'Li1.3Al0.3Ti1.7(PO4)3'."
    ),
    instruction="""You are the Context Librarian for a materials science research group.
    Your job is to read the introduction and methods sections of solid-state electrolyte papers.

    WORKFLOW:
    1. Receive the combined text of the introduction and methods sections.
    2. Call `retrieve_document_abbreviations` on that text.
    3. Return the full abbreviation map as a JSON dict so other agents can use it.

    Be thorough. Look for polymer abbreviations (PEO, PVDF, PMMA), ceramic abbreviations
    (LLZO, LAGP, LATP, NASICON), composite labels (CPE, SPE), and loading-based names
    like "10 wt% LLZO in PEO" that might be abbreviated as "C10" or "CPE10".
    """,
    tools=[retrieve_document_abbreviations],
)

# =============================================================================
# 2. The Data Extractor
# =============================================================================
extractor_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="data_extractor",
    description=(
        "Extracts raw conductivity data points from figures, tables, and text. "
        "Route to this agent when you have a specific figure path, markdown table, "
        "or text paragraph that needs to be parsed for ionic conductivity numbers."
    ),
    instruction="""You are the Data Extractor specializing in solid-state electrolyte literature.
    You read ionic conductivity plots (Arrhenius curves, σ vs T) and Markdown tables.

    WORKFLOW:
    - For a figure: call `extract_from_plot(image_path, caption)`. The function handles
      subplot detection, cropping, and axis-aware extraction automatically.
    - For a markdown table: call `extract_from_markdown_table(table_markdown, caption)`.
    - Each call returns a list of measurement dicts with raw and normalized values.

    RULES:
    - DO NOT attempt to normalize or convert units yourself. The tools do this.
    - DO NOT resolve material names. Report exact labels as extracted (e.g., 'LATP03', 'x=0.2').
    - If a figure contains no conductivity data (e.g., XRD, SEM, Nyquist plots), report "no conductivity data found" for that figure.
    - Collect ALL returned measurements across all assets and pass them upstream.
    """,
    tools=[extract_from_plot, extract_from_markdown_table],
)

# =============================================================================
# 3. The Materials Chemist
# =============================================================================
chemist_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="materials_chemist",
    description=(
        "Handles chemical validation, stoichiometry checks, and physical bounds verification. "
        "Route to this agent with the full list of raw measurements and the abbreviation map. "
        "Returns the same list enriched with canonical_formula, reduced_formula, "
        "normalized values, and confidence flags."
    ),
    instruction="""You are the Materials Chemist responsible for Quality Assurance on solid-state electrolyte data.

    WORKFLOW:
    1. Receive the full list of raw measurements (from the extractor) and the abbreviation map (from the librarian).
    2. Call `QA_process_measurement_batch(raw_measurements, abbreviation_map)` once with the entire batch.
       This single call runs name resolution, stoichiometry validation, and bounds checking for every
       measurement in Python — no looping required from your side.
    3. Review the returned list. Your job is to interpret the QA report:
       - Identify any measurements where canonical_formula is None (unresolvable deterministically).
         Report these to the lead researcher for LLM-based canonicalization.
       - Summarize patterns in the warnings (e.g., "8 measurements flagged for conductivity > 0.5 S/cm,
         likely a unit mismatch in Figure 3").
       - Do NOT discard flagged measurements — annotate and return them all.
    4. Return the processed batch plus your summary of issues found.
    """,
    tools=[QA_process_measurement_batch],
)

# =============================================================================
# 4. The Lead Researcher (Root Agent)
# =============================================================================
lead_researcher_agent = LlmAgent(
    model="gemini-2.5-pro",
    name="lead_researcher_pm",
    description="The main orchestrator for the ionic conductivity data extraction pipeline.",
    instruction="""You are the Lead Researcher coordinating an automated data extraction pipeline
    for solid-state electrolyte ionic conductivity literature.

    YOUR WORKFLOW:
    1. PARSE: Call `parse_markdown_structure(file_content, asset_dir)` with the full markdown
       text AND the asset directory path. This returns sections, images (each with a 'full_path'
       field), and markdown tables.

    2. CONTEXT: Identify intro and methods section text. Delegate to `context_librarian`
       to build the paper-wide abbreviation map. Ask it to process the combined intro+methods text.

    3. EXTRACT: For every image in the list from step 1, delegate to `data_extractor` using
       the image's 'full_path' value (not just the filename) and its caption.
       For every markdown table, delegate similarly.
       Share the abbreviation map in your delegation message as context.

    4. VALIDATE: Pass ALL extracted raw measurements and the abbreviation map to
       `materials_chemist` for stoichiometry validation, name resolution, and bounds checking.

    5. COMPILE: Review the validated measurements. Apply these rules:
       - Discard measurements where normalized_conductivity is None (σ₀, pre-exponential).
       - For duplicate (composition, temperature) pairs prefer: figure > table > text,
         and measured > cited data.
       - Output a final JSON report with all validated measurements.

    FINAL OUTPUT FORMAT:
    Return a JSON object:
    {
      "doc_title": "...",
      "total_measurements": N,
      "measurements": [
        {
          "raw_composition": "...",
          "canonical_formula": "...",  // resolved or null
          "material_class": "Polymer|Ceramic|Composite",
          "normalized_conductivity": 1.2e-4,  // S/cm
          "normalized_temperature_c": 25.0,
          "source_type": "figure|table|text",
          "confidence": "high|low",
          "warnings": [...]
        }
      ]
    }

    Rely on your sub-agents for all parsing, extraction, and validation work.
    """,
    tools=[parse_markdown_structure],
    sub_agents=[librarian_agent, extractor_agent, chemist_agent],
)

# =============================================================================
# Execution
# =============================================================================

async def run_extraction(markdown_file_path: str) -> str:
    """
    Run the full extraction pipeline on a single markdown paper file.

    Args:
        markdown_file_path: Path to the .md file produced by the PDF converter.

    Returns:
        The agent's final JSON report as a string.
    """
    from pathlib import Path
    md_path = Path(markdown_file_path).resolve()
    asset_dir = str(md_path.parent)  # images live alongside the .md file

    with open(md_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=lead_researcher_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=APP_NAME, user_id="researcher"
    )

    prompt = (
        f"Please run the full data extraction pipeline on this paper.\n"
        f"Markdown file : {md_path}\n"
        f"Asset directory (images): {asset_dir}\n\n"
        f"Full document content:\n\n{file_content[:50000]}"
    )

    final_response = ""
    async for event in runner.run_async(
        user_id="researcher",
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        ),
    ):
        _log_event(event)

        # Only capture the root agent's final text response, not sub-agent turns
        if (
            event.is_final_response()
            and event.author == lead_researcher_agent.name
            and event.content
            and event.content.parts
        ):
            text_parts = [p.text for p in event.content.parts if p.text]
            if text_parts:
                final_response = "\n".join(text_parts)

    return final_response


def _log_event(event) -> None:
    """Print a one-line summary of every ADK event for debugging."""
    author = getattr(event, "author", "?")
    is_final = event.is_final_response()

    tool_calls = event.get_function_calls()
    tool_responses = event.get_function_responses()

    if tool_calls:
        for tc in tool_calls:
            args_preview = str(tc.args)[:120].replace("\n", " ")
            print(f"  [TOOL CALL ] {author} → {tc.name}({args_preview})")
    elif tool_responses:
        for tr in tool_responses:
            resp_preview = str(tr.response)[:120].replace("\n", " ")
            print(f"  [TOOL RESP ] {author} ← {tr.name}: {resp_preview}")
    elif event.content and event.content.parts:
        text_parts = [p.text for p in event.content.parts if p.text]
        text_preview = " ".join(text_parts)[:200].replace("\n", " ")
        marker = "✅ FINAL" if is_final else "      ..."
        print(f"  [{marker}] {author}: {text_preview}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_paper.md>")
        print("Example: python agent.py ./papers/composite_electrolyte_2025.md")
        sys.exit(1)

    document_path = sys.argv[1]
    result = asyncio.run(run_extraction(document_path))
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(result)
