"""Quick test: run the verifier multi-agent system on the LLZO nanowires paper."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verifier_agent import run_verifier

PAPER_DIR = Path(
    "/Users/bourn23/Downloads/general/PageIndex/output/downselectedpapers_jiyoung/"
    "Composite Polymer Electrolytes with Li7La3Zr2O12 Garnet-Type Nanowires as Ceramic Fillers"
)

LOG_PATH    = PAPER_DIR / "extraction_log.jsonl"
RESULTS     = PAPER_DIR / "robust_results_v8_verifier_test.json"
PDF_PATH    = PAPER_DIR / (
    "Composite Polymer Electrolytes with Li7La3Zr2O12 "
    "Garnet-Type Nanowires as Ceramic Fillers.pdf"
)

async def main():
    print(f"Log:     {LOG_PATH}")
    print(f"Results: {RESULTS}")
    print(f"PDF:     {PDF_PATH}")
    print("-" * 60)
    report = await run_verifier(
        log_path=str(LOG_PATH),
        results_json_path=str(RESULTS),
        pdf_path=str(PDF_PATH),
    )
    print("\n=== VERIFIER FINAL REPORT ===")
    print(report)

asyncio.run(main())
