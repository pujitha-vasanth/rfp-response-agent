"""
Agent 2: Matcher.

For each requirement, searches the capability library for evidence and decides
whether we meet it fully, partially or not at all.
"""

import json

from rfp_agent import config, demo_llm, knowledge
from rfp_agent.llm import ask_json, run_in_parallel

SYSTEM_PROMPT = f"""You are the Matcher agent in a bid team for {config.COMPANY_NAME}. For one RFP requirement you decide how well {config.COMPANY_NAME} meets it, using ONLY the capability entries provided. Do not use outside knowledge and do not assume features that are not written in the entries.

Compliance levels:
- "Full": the entries show we meet every part of the requirement as written.
- "Partial": we meet only part of it; or we meet it only through an add-on, extra cost, custom development or a workaround; or we offer the same thing at a lower level than required (for example a lower SLA or a lower tested capacity).
- "None": no entry covers it, or an entry says we do not do it.

Check numbers carefully (percentages, hours, counts, weeks). Offering more than required still meets it, for example 5,000 tested when 2,000 is required. For some numbers lower is better, such as response times, RPO, RTO and weeks to go live.
Always read the "limits" of each entry. They often decide between Full, Partial and None.

Return JSON in this shape:
{{"status": "Full", "evidence_ids": ["CAP-01"], "reason": "one sentence explaining the decision"}}
evidence_ids are the entries that support your decision. Leave the list empty if nothing relevant was found."""

VALID_STATUSES = {"full": "Full", "partial": "Partial", "none": "None"}


def match_requirement(requirement: dict) -> dict:
    candidates = knowledge.search(requirement["text"], top_k=4)
    entries = [{k: c[k] for k in ("id", "title", "statement", "limits")} for c in candidates]
    user_prompt = (
        f"Requirement ({requirement['priority']}): {requirement['text']}\n\n"
        f"Capability entries:\n{json.dumps(entries, indent=2)}"
    )
    result = ask_json(SYSTEM_PROMPT, user_prompt, demo_answer=lambda: demo_llm.match(requirement, candidates))

    status = VALID_STATUSES.get(str(result.get("status", "")).strip().lower(), "None")
    candidate_ids = {c["id"] for c in candidates}
    evidence_ids = [i for i in result.get("evidence_ids", []) if i in candidate_ids]  # ignore made-up ids
    return {"status": status, "evidence_ids": evidence_ids, "reason": str(result.get("reason", ""))}


def match_all(requirements: list[dict]) -> dict:
    results = run_in_parallel(match_requirement, requirements)
    return {req["id"]: result for req, result in zip(requirements, results)}
