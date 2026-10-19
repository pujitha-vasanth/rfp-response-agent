"""
Scores the agent team against hand-labelled answers.

For each sample RFP in data/rfps/ there is a file in data/golden/ listing every
real requirement and the correct compliance level (Full, Partial or None).
This script runs the agents on each RFP and compares.

Usage (from the project folder):
    python evaluate.py

What the numbers mean:
    Requirements found   share of real requirements the Reader extracted
    Extra items          things the Reader extracted that are not real requirements
    Compliance accuracy  share of found requirements where the Matcher chose the right level
    Gaps caught          share of real gaps (Partial or None) the Matcher flagged as a gap
    Overclaims           real gaps the Matcher marked as Full. This is the costly mistake:
                         it puts a promise in the bid that the company cannot keep.
    First-pass rate      share of drafts the Reviewer passed without a rewrite
"""

import json
import time

from rfp_agent import config, llm
from rfp_agent.documents import read_rfp
from rfp_agent.graph import run_pipeline
from rfp_agent.knowledge import tokenize

MATCH_THRESHOLD = 0.6  # share of a golden requirement's words that must appear in an extracted one


def pair_requirements(golden: list[dict], extracted: list[dict]) -> dict:
    """Pair each golden requirement with the extracted requirement that covers most of its words."""
    pairs, used = {}, set()
    for g in golden:
        g_words = set(tokenize(g["text"]))
        best_id, best_overlap = None, 0.0
        for e in extracted:
            if e["id"] in used:
                continue
            overlap = len(g_words & set(tokenize(e["text"]))) / max(len(g_words), 1)
            if overlap > best_overlap:
                best_id, best_overlap = e["id"], overlap
        if best_id and best_overlap >= MATCH_THRESHOLD:
            pairs[g["id"]] = best_id
            used.add(best_id)
    return pairs


def evaluate_rfp(rfp_path, golden_path) -> dict:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    llm.reset_stats()
    started = time.time()
    state = run_pipeline(read_rfp(rfp_path))
    seconds = time.time() - started

    extracted = state["requirements"]
    matches = state["matches"]
    answers = state.get("answers", {})
    pairs = pair_requirements(golden, extracted)

    correct, overclaims, real_gaps, gaps_caught, mistakes = 0, 0, 0, 0, []
    for g in golden:
        is_gap = g["expected_status"] != "Full"
        real_gaps += is_gap
        if g["id"] not in pairs:
            mistakes.append({"golden_id": g["id"], "problem": "not extracted", "requirement": g["text"]})
            continue
        predicted = matches[pairs[g["id"]]]["status"]
        correct += predicted == g["expected_status"]
        gaps_caught += is_gap and predicted != "Full"
        overclaims += is_gap and predicted == "Full"
        if predicted != g["expected_status"]:
            mistakes.append({"golden_id": g["id"], "problem": f"expected {g['expected_status']}, got {predicted}",
                             "requirement": g["text"]})

    first_pass = sum(a["drafts"] == 1 and a["passed"] for a in answers.values())
    return {
        "rfp": rfp_path.name,
        "golden_requirements": len(golden),
        "extracted": len(extracted),
        "requirements_found": len(pairs) / len(golden),
        "extra_items": len(extracted) - len(pairs),
        "compliance_accuracy": correct / max(len(pairs), 1),
        "gaps_caught": gaps_caught / max(real_gaps, 1),
        "overclaims": overclaims,
        "answers_drafted": len(answers),
        "first_pass_rate": first_pass / max(len(answers), 1),
        "answers_rewritten": sum(a["drafts"] > 1 for a in answers.values()),
        "answers_needing_human_edit": sum(not a["passed"] for a in answers.values()),
        "model_calls": llm.stats["calls"],
        "seconds": round(seconds, 1),
        "mistakes": mistakes,
    }


def main():
    print(f"\nModel provider: {config.LLM_PROVIDER}")
    if config.LLM_PROVIDER == "demo":
        print("Note: demo mode uses simple rules, so these scores are NOT the agents' real quality.")

    results = []
    for golden_path in sorted(config.GOLDEN_DIR.glob("*.json")):
        rfp_path = config.SAMPLE_RFP_DIR / f"{golden_path.stem}.pdf"
        print(f"\nEvaluating {rfp_path.name} ...")
        results.append(evaluate_rfp(rfp_path, golden_path))

    print("\n" + "-" * 78)
    print(f"{'Metric':<32}" + "".join(f"{r['rfp'][:22]:>23}" for r in results))
    print("-" * 78)
    rows = [
        ("Requirements found", "requirements_found", "pct"),
        ("Extra items extracted", "extra_items", "int"),
        ("Compliance accuracy", "compliance_accuracy", "pct"),
        ("Gaps caught", "gaps_caught", "pct"),
        ("Overclaims (gap marked Full)", "overclaims", "int"),
        ("Reviewer first-pass rate", "first_pass_rate", "pct"),
        ("Answers rewritten", "answers_rewritten", "int"),
        ("Answers needing human edit", "answers_needing_human_edit", "int"),
        ("Model calls", "model_calls", "int"),
        ("Seconds", "seconds", "int"),
    ]
    for label, key, kind in rows:
        values = [f"{r[key]:.0%}" if kind == "pct" else f"{r[key]}" for r in results]
        print(f"{label:<32}" + "".join(f"{v:>23}" for v in values))
    print("-" * 78)

    for r in results:
        if r["mistakes"]:
            print(f"\nMistakes in {r['rfp']}:")
            for m in r["mistakes"]:
                print(f"  {m['golden_id']}: {m['problem']}  |  {m['requirement'][:70]}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    report_path = config.OUTPUT_DIR / f"eval_report_{config.LLM_PROVIDER}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report saved to {report_path.relative_to(config.PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
