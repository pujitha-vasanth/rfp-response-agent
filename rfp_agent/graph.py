"""
Wires the four agents together with LangGraph.

    read -> match -> write -> review -> finish
                               ^         |
                               |_________|  back to write if a draft failed review

The "state" is one dictionary that every step reads from and adds to.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from rfp_agent import config
from rfp_agent.agents import matcher, reader, reviewer, writer
from rfp_agent.llm import run_in_parallel


class RFPState(TypedDict, total=False):
    rfp_text: str
    requirements: list[dict]        # from the Reader
    matches: dict                   # requirement id -> {status, evidence_ids, reason}
    answers: dict                   # requirement id -> {text, score, feedback, drafts, passed, reviewed}
    log: Annotated[list[str], operator.add]   # progress messages; each step adds to the list


# ---------- Steps (nodes) ----------

def read_step(state: RFPState) -> dict:
    requirements = reader.extract_requirements(state["rfp_text"])
    mandatory = sum(r["priority"] == "Mandatory" for r in requirements)
    return {
        "requirements": requirements,
        "log": [f"Reader: found {len(requirements)} requirements ({mandatory} mandatory)."],
    }


def match_step(state: RFPState) -> dict:
    matches = matcher.match_all(state["requirements"])
    counts = {s: sum(m["status"] == s for m in matches.values()) for s in ("Full", "Partial", "None")}
    return {
        "matches": matches,
        "log": [f"Matcher: {counts['Full']} full, {counts['Partial']} partial, {counts['None']} not met."],
    }


def write_step(state: RFPState) -> dict:
    answers = dict(state.get("answers", {}))
    matches = state["matches"]

    # Write a first draft for every requirement we can answer,
    # and rewrite any draft the Reviewer did not pass.
    to_write = []
    for req in state["requirements"]:
        if matches[req["id"]]["status"] == "None":
            continue  # nothing to claim: this goes to a human as a gap
        current = answers.get(req["id"])
        if current is None or (not current["passed"] and current["drafts"] < config.MAX_DRAFTS):
            to_write.append(req)

    def write_one(req):
        current = answers.get(req["id"])
        return writer.write_answer(
            req, matches[req["id"]],
            previous_draft=current["text"] if current else None,
            feedback=current["feedback"] if current else None,
        )

    new_texts = run_in_parallel(write_one, to_write)

    rewrites = 0
    for req, text in zip(to_write, new_texts):
        previous = answers.get(req["id"])
        rewrites += previous is not None
        answers[req["id"]] = {
            "text": text,
            "score": None,
            "feedback": "",
            "drafts": (previous["drafts"] if previous else 0) + 1,
            "passed": False,
            "reviewed": False,
        }

    message = f"Writer: rewrote {rewrites} answers using reviewer feedback." if rewrites else f"Writer: drafted {len(to_write)} answers."
    return {"answers": answers, "log": [message]}


def review_step(state: RFPState) -> dict:
    answers = {k: dict(v) for k, v in state["answers"].items()}
    requirements = {r["id"]: r for r in state["requirements"]}
    to_review = [rid for rid, a in answers.items() if not a["reviewed"]]

    results = run_in_parallel(
        lambda rid: reviewer.review_answer(requirements[rid], state["matches"][rid], answers[rid]["text"]),
        to_review,
    )
    for rid, result in zip(to_review, results):
        answers[rid].update(result, reviewed=True)

    failed = sum(not answers[rid]["passed"] for rid in to_review)
    return {"answers": answers, "log": [f"Reviewer: checked {len(to_review)} drafts, {failed} sent back."]}


def finish_step(state: RFPState) -> dict:
    answers = state.get("answers", {})
    stuck = sum(not a["passed"] for a in answers.values())
    gaps = sum(m["status"] != "Full" for m in state["matches"].values())
    return {"log": [f"Done: {gaps} requirements need a bid decision, {stuck} answers need a human edit."]}


# ---------- Routing ----------

def after_review(state: RFPState) -> str:
    """Go back to the Writer if any draft failed and still has tries left."""
    for answer in state["answers"].values():
        if not answer["passed"] and answer["drafts"] < config.MAX_DRAFTS:
            return "write"
    return "finish"


# ---------- Build the graph ----------

def build_graph():
    graph = StateGraph(RFPState)
    graph.add_node("read", read_step)
    graph.add_node("match", match_step)
    graph.add_node("write", write_step)
    graph.add_node("review", review_step)
    graph.add_node("finish", finish_step)

    graph.add_edge(START, "read")
    graph.add_edge("read", "match")
    graph.add_edge("match", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", after_review, {"write": "write", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def run_pipeline(rfp_text: str, on_progress=None) -> RFPState:
    """
    Run the whole team on one RFP and return the final state.

    on_progress is an optional function that gets each log message as it
    happens (the command line prints it, the web app shows it on screen).
    """
    app = build_graph()
    final_state = None
    for mode, data in app.stream({"rfp_text": rfp_text, "log": []}, stream_mode=["updates", "values"]):
        if mode == "updates" and on_progress:
            for step_output in data.values():
                for message in (step_output or {}).get("log", []):
                    on_progress(message)
        elif mode == "values":
            final_state = data
    return final_state
