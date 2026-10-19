"""
Demo mode: simple rules that stand in for the AI model.

This lets you run the whole system with no API key, for example to check your
setup or to show the flow in an interview without internet. The answers are
much weaker than a real model's. Use LLM_PROVIDER=azure or openai for real results.

Each function here mirrors one agent and returns the same JSON shape the
real model would return.
"""

import re

from rfp_agent import config
from rfp_agent.knowledge import tokenize

HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
REQUIREMENT_WORD = re.compile(r"\b(must|shall|should)\b", re.IGNORECASE)
PROCESS_WORDS = ("bidder", "tender", "proposal", "portal", "submitted", "late ", "clarification", "marked", "shown as")
NEGATIVE_LIMITS = ("do not", "is not supported", "not supported", "there is no", "not pci", "not hold")


def extract(chunk_text: str) -> dict:
    """Reader stand-in: keep sentences that contain must / shall / should."""
    sections = []  # list of [heading, text]
    current = ["General", ""]
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADING.match(line) and len(line) < 80:
            sections.append(current)
            current = [line, ""]
        else:
            current[1] += " " + line
    sections.append(current)

    found = []
    for heading, text in sections:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
        for sentence in sentences:
            sentence = re.sub(r"^(MUST|SHOULD)\s*:\s*", "", sentence.strip("• ")).strip()
            if not REQUIREMENT_WORD.search(sentence):
                continue
            if any(word in sentence.lower() for word in PROCESS_WORDS):
                continue
            mandatory = re.search(r"\b(must|shall)\b", sentence, re.IGNORECASE)
            found.append({
                "section": heading,
                "text": sentence.strip(),
                "priority": "Mandatory" if mandatory else "Desirable",
            })
    return {"requirements": found}


def match(requirement: dict, candidates: list[dict]) -> dict:
    """Matcher stand-in: use the keyword score and the limits text."""
    top = candidates[0]
    req_tokens = set(tokenize(requirement["text"]))
    # Words that appear in the limits but not in the statement point to a gap.
    limit_only_tokens = set(tokenize(top["limits"])) - set(tokenize(top["statement"]))
    shared = {t for t in req_tokens & limit_only_tokens if len(t) > 3}

    if top["limits"] and shared:
        negative = any(phrase in top["limits"].lower() for phrase in NEGATIVE_LIMITS)
        status = "None" if negative else "Partial"
    elif top["score"] >= 8:
        status = "Full"
    elif top["score"] >= 4:
        status = "Partial"
    else:
        status = "None"

    evidence = [top["id"]] if status != "None" or top["score"] >= 4 else []
    return {"status": status, "evidence_ids": evidence, "reason": f"Demo rule based on keyword score {top['score']}."}


def write(status: str, evidence: list[dict], feedback: str | None) -> dict:
    """Writer stand-in: stitch the evidence statements together.

    On the first draft of a Partial answer it leaves out the limits on purpose,
    so you can watch the Reviewer send it back for a rewrite.
    """
    company = config.COMPANY_NAME
    opening = f"{company} fully meets this requirement." if status == "Full" else f"{company} partially meets this requirement."
    body = " ".join(e["statement"] for e in evidence[:2])
    answer = f"{opening} {body}"
    if feedback and status == "Partial":
        limits = " ".join(e["limits"] for e in evidence if e["limits"])
        answer += f" Please note: {limits or 'part of this requirement would need to be agreed during contracting.'}"
    return {"answer": answer}


def review(status: str, answer: str) -> dict:
    """Reviewer stand-in: a Partial answer must say what is not covered."""
    if status == "Partial" and "Please note" not in answer:
        return {"score": 3, "feedback": "The answer does not say which part of the requirement is not met. Add the limitation from the evidence."}
    return {"score": 5, "feedback": "None"}
