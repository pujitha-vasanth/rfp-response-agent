"""
Run the RFP response team from the terminal.

Usage (from the project folder):
    python run.py data/rfps/harbourview_logistics_rfp.pdf

Outputs go to the output/ folder:
    <rfp name>_compliance_matrix.xlsx
    <rfp name>_draft_response.docx
"""

import sys
import time
from pathlib import Path

from rfp_agent import config, llm
from rfp_agent.documents import read_rfp
from rfp_agent.graph import run_pipeline
from rfp_agent.outputs import build_rows, export_excel, export_word, pretty_name


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <path to RFP file (.pdf, .docx or .txt)>")
        print("Example: python run.py data/rfps/harbourview_logistics_rfp.pdf")
        sys.exit(1)

    rfp_path = Path(sys.argv[1])
    if not rfp_path.exists():
        print(f"File not found: {rfp_path}")
        sys.exit(1)

    print(f"\nRFP: {rfp_path.name}")
    print(f"Model provider: {config.LLM_PROVIDER}\n")

    started = time.time()
    state = run_pipeline(read_rfp(rfp_path), on_progress=lambda message: print(f"  {message}"))
    rows = build_rows(state)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    excel_path = config.OUTPUT_DIR / f"{rfp_path.stem}_compliance_matrix.xlsx"
    word_path = config.OUTPUT_DIR / f"{rfp_path.stem}_draft_response.docx"
    export_excel(rows, excel_path)
    export_word(rows, pretty_name(rfp_path.stem), word_path)

    print("\nRequirements that need a human decision:")
    for row in rows:
        if row["human_action"]:
            print(f"  {row['id']} [{row['status']}] {row['requirement'][:80]}")

    print(f"\nFinished in {time.time() - started:.0f} seconds, {llm.stats['calls']} model calls.")
    print(f"Compliance matrix: {excel_path.relative_to(config.PROJECT_ROOT)}")
    print(f"Draft response:    {word_path.relative_to(config.PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
