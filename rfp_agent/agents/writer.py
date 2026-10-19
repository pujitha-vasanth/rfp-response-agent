"""
Agent 3: Writer.

Drafts the proposal answer for one requirement, using only the evidence the
Matcher found. When the Reviewer sends a draft back, the Writer gets the old
draft plus the feedback and rewrites it.
"""

import json

from rfp_agent import config, demo_llm, knowledge
from rfp_agent.llm import ask_json

COMPANY = config.COMPANY_NAME

SYSTEM_PROMPT = f"""You are the Writer agent in a bid team for {COMPANY}. You write the answer to one RFP requirement for the formal proposal.

Rules:
- Use ONLY facts found in the evidence. Never invent numbers, certifications, customer names, features or commitments.
- Open with a clear compliance statement: "{COMPANY} fully meets this requirement." or "{COMPANY} partially meets this requirement."
- For a partial answer, say exactly which part is met, which part is not, and any alternative the evidence offers (an add-on, custom work, extra cost).
- Speak to the buyer's need, not just our features. Refer to the specifics in the requirement (numbers, systems, channels).
- 50 to 130 words. Plain professional English. No marketing language, no bullet points, no headings, no em dashes.
- If you are given reviewer feedback on an earlier draft, fix every point it raises.

Return JSON in this shape:
{{"answer": "..."}}"""


def write_answer(requirement: dict, match: dict, previous_draft: str | None = None, feedback: str | None = None) -> str:
    evidence = knowledge.get_by_ids(match["evidence_ids"])
    evidence_for_prompt = [{k: e[k] for k in ("id", "title", "statement", "limits")} for e in evidence]

    user_prompt = (
        f"Requirement ({requirement['priority']}): {requirement['text']}\n"
        f"Compliance level decided by the Matcher: {match['status']}\n"
        f"Matcher's reason: {match['reason']}\n\n"
        f"Evidence:\n{json.dumps(evidence_for_prompt, indent=2)}"
    )
    if previous_draft and feedback:
        user_prompt += f"\n\nYour previous draft:\n{previous_draft}\n\nReviewer feedback to fix:\n{feedback}"

    result = ask_json(
        SYSTEM_PROMPT, user_prompt,
        demo_answer=lambda: demo_llm.write(match["status"], evidence, feedback),
    )
    return str(result.get("answer", "")).strip()
