"""
Agent 4: Reviewer.

Checks each draft answer against the requirement and the evidence, like a bid
manager would. It scores the draft from 1 to 5. Drafts below the pass score
go back to the Writer with specific feedback.

The Reviewer never sees the Writer's instructions, only the draft. That keeps
it an independent check rather than the Writer marking its own work.
"""

import json

from rfp_agent import config, demo_llm, knowledge
from rfp_agent.llm import ask_json

SYSTEM_PROMPT = f"""You are the Reviewer agent in a bid team for {config.COMPANY_NAME}. You check one draft RFP answer before it goes to the buyer. You did not write it. Be strict: a wrong claim in a bid can become a contractual commitment.

Score the draft from 1 to 5:
5 = ready to send.
4 = only minor wording issues.
3 = a real problem: misses part of the requirement, is vague, or the compliance statement is unclear.
2 = makes a claim that the evidence does not support, or overstates compliance.
1 = wrong or off topic.

Check:
1. Does it answer every part of the requirement?
2. Is every factual claim (numbers, certifications, features, timelines) supported by the evidence? Anything not in the evidence is unsupported.
3. Does the compliance statement match the compliance level given? For Partial, does it say clearly what is not met?
4. Is it clear, specific and between 50 and 130 words?

Return JSON in this shape:
{{"score": 4, "feedback": "specific instructions to fix the draft, or None if the score is 5"}}"""


def review_answer(requirement: dict, match: dict, answer: str) -> dict:
    evidence = knowledge.get_by_ids(match["evidence_ids"])
    evidence_for_prompt = [{k: e[k] for k in ("id", "statement", "limits")} for e in evidence]
    user_prompt = (
        f"Requirement ({requirement['priority']}): {requirement['text']}\n"
        f"Compliance level: {match['status']}\n\n"
        f"Evidence:\n{json.dumps(evidence_for_prompt, indent=2)}\n\n"
        f"Draft answer:\n{answer}"
    )
    result = ask_json(SYSTEM_PROMPT, user_prompt, demo_answer=lambda: demo_llm.review(match["status"], answer))

    try:
        score = max(1, min(5, int(result.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    return {
        "score": score,
        "feedback": str(result.get("feedback", "")),
        "passed": score >= config.PASS_SCORE,
    }
