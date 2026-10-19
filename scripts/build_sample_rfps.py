"""
Turns the sample RFP text files in data/rfps/source/ into PDF files in data/rfps/.

You only need this if you edit the text files. The PDFs are already in the repo.

Simple markup used in the text files:
    TITLE: ...       big title on the first page
    SUBTITLE: ...    line under the title
    # Heading        section heading
    ## Heading       sub-section heading
    - item           bullet point
    anything else    normal paragraph

Run from the project folder:
    python scripts/build_sample_rfps.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, ListFlowable, ListItem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "rfps" / "source"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rfps"

styles = getSampleStyleSheet()
STYLE_TITLE = ParagraphStyle("RfpTitle", parent=styles["Title"], fontSize=20, spaceAfter=6)
STYLE_SUBTITLE = ParagraphStyle("RfpSubtitle", parent=styles["Normal"], fontSize=10, textColor="#555555", spaceAfter=18)
STYLE_H1 = ParagraphStyle("RfpH1", parent=styles["Heading1"], fontSize=14, spaceBefore=12)
STYLE_H2 = ParagraphStyle("RfpH2", parent=styles["Heading2"], fontSize=12, spaceBefore=8)
STYLE_BODY = ParagraphStyle("RfpBody", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)


def build_pdf(text_file: Path, pdf_file: Path) -> None:
    story = []
    bullets = []

    def flush_bullets():
        # Bullets are collected first so they can be drawn as one list.
        if bullets:
            items = [ListItem(Paragraph(b, STYLE_BODY), leftIndent=12) for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="•", bulletFontName="Symbol", leftIndent=14))
            story.append(Spacer(1, 6))
            bullets.clear()

    for raw_line in text_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            bullets.append(line[2:])
            continue
        flush_bullets()
        if line.startswith("TITLE:"):
            story.append(Paragraph(line[6:].strip(), STYLE_TITLE))
        elif line.startswith("SUBTITLE:"):
            story.append(Paragraph(line[9:].strip(), STYLE_SUBTITLE))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], STYLE_H2))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], STYLE_H1))
        else:
            story.append(Paragraph(line, STYLE_BODY))
    flush_bullets()

    doc = SimpleDocTemplate(
        str(pdf_file), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    doc.build(story)


if __name__ == "__main__":
    for text_file in sorted(SOURCE_DIR.glob("*.txt")):
        pdf_file = OUTPUT_DIR / f"{text_file.stem}.pdf"
        build_pdf(text_file, pdf_file)
        print(f"Built {pdf_file.relative_to(PROJECT_ROOT)}")
