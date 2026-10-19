"""
Agent 1: Reader.

Reads the RFP and pulls out every requirement as a separate item, with its
section and whether it is mandatory or desirable.
"""

import re

from rfp_agent import demo_llm
from rfp_agent.llm import ask_json, run_in_parallel

SYSTEM_PROMPT = """You are the Reader agent in a bid team. You read part of a Request for Proposal (RFP) and list every requirement that the bidder's product, service or company must meet.

Rules:
- Include anything the buyer requires or would like from the solution or the supplier: security, data protection, features, channels, integrations, service levels, capacity, support, delivery, training, commercial terms, insurance, certifications, references.
- Do NOT include instructions about the tender process itself: deadlines, how or where to submit, who to contact, how bids are scored, or background facts about the buyer.
- One requirement per item. If one sentence contains two separate requirements, split it into two items.
- Keep the buyer's wording as closely as possible, but make each item understandable on its own (replace "it" or "this" with what it refers to).
- priority: "Mandatory" for must, shall, MUST or required. "Desirable" for should, SHOULD, desirable or preferred.
- section: the nearest section heading above the requirement, for example "4.1 Hosting, security and data protection".

Return JSON in this shape:
{"requirements": [{"section": "...", "text": "...", "priority": "Mandatory"}]}
If this part of the RFP has no requirements, return {"requirements": []}."""


def split_into_chunks(text: str, max_chars: int = 6000) -> list[str]:
    """Split long RFPs into chunks at paragraph breaks, so each model call stays small."""
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        if len(current) + len(paragraph) > max_chars and current:
            chunks.append(current)
            current = ""
        current += paragraph + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def _extract_from_chunk(chunk: str) -> list[dict]:
    result = ask_json(
        SYSTEM_PROMPT,
        f"RFP text:\n\n{chunk}",
        demo_answer=lambda: demo_llm.extract(chunk),
    )
    return result.get("requirements", [])


def extract_requirements(rfp_text: str) -> list[dict]:
    chunk_results = run_in_parallel(_extract_from_chunk, split_into_chunks(rfp_text))

    requirements, seen = [], set()
    for chunk_requirements in chunk_results:
        for item in chunk_requirements:
            text = str(item.get("text", "")).strip(" •-*\t\n")
            key = re.sub(r"[^a-z0-9]", "", text.lower())
            if not text or key in seen:  # skip empty items and duplicates
                continue
            seen.add(key)
            priority = "Desirable" if str(item.get("priority", "")).lower().startswith("d") else "Mandatory"
            requirements.append({
                "id": f"R{len(requirements) + 1:03d}",
                "section": str(item.get("section", "General")).strip() or "General",
                "text": text,
                "priority": priority,
            })
    return requirements
