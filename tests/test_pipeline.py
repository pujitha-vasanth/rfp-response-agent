"""
Quick checks that the system works end to end. No API key needed.

Run from the project folder:
    pytest
"""

import json
from types import SimpleNamespace

import pytest

from rfp_agent import config, knowledge, llm
from rfp_agent.agents.reader import split_into_chunks
from rfp_agent.documents import read_rfp
from rfp_agent.graph import run_pipeline
from rfp_agent.outputs import build_rows, export_excel, export_word

HARBOURVIEW = config.SAMPLE_RFP_DIR / "harbourview_logistics_rfp.pdf"
ALDERMOOR = config.SAMPLE_RFP_DIR / "aldermoor_university_rfp.pdf"


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    """Every test runs in demo mode, whatever is in your .env file."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "demo")


def test_search_finds_the_right_capability():
    assert knowledge.search("Is the supplier PCI DSS certified?")[0]["id"] == "CAP-05"
    assert knowledge.search("WCAG 2.2 AA accessibility")[0]["id"] == "CAP-19"


def test_long_text_is_split_into_chunks():
    text = "\n".join(["A paragraph of RFP text."] * 2000)
    chunks = split_into_chunks(text, max_chars=6000)
    assert len(chunks) > 1
    assert all(len(c) <= 6100 for c in chunks)


def test_full_pipeline_and_exports(tmp_path):
    state = run_pipeline(read_rfp(HARBOURVIEW))
    assert len(state["requirements"]) >= 20
    assert set(state["matches"]) == {r["id"] for r in state["requirements"]}

    rows = build_rows(state)
    export_excel(rows, tmp_path / "matrix.xlsx")
    export_word(rows, "Test RFP", tmp_path / "response.docx")
    assert (tmp_path / "matrix.xlsx").stat().st_size > 0
    assert (tmp_path / "response.docx").stat().st_size > 0


def test_reviewer_sends_weak_drafts_back():
    state = run_pipeline(read_rfp(ALDERMOOR))
    answers = state["answers"].values()
    assert any(a["drafts"] > 1 for a in answers), "expected at least one rewrite"
    assert all(a["drafts"] <= config.MAX_DRAFTS for a in answers)


def test_real_model_path_with_a_stub(monkeypatch):
    """Checks the non-demo code path (prompts, JSON parsing) using a stand-in client."""

    def fake_create(model, messages, response_format):
        system = messages[0]["content"]
        if "Reader agent" in system:
            body = {"requirements": [{"section": "4.1", "text": "The supplier must hold ISO 27001.", "priority": "Mandatory"}]}
        elif "Matcher agent" in system:
            body = {"status": "full", "evidence_ids": ["CAP-05", "CAP-999"], "reason": "Certified."}
        elif "Writer agent" in system:
            body = {"answer": "Brightpath Digital fully meets this requirement."}
        else:
            body = {"score": 5, "feedback": "None"}
        message = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    stub_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm, "_get_client", lambda: stub_client)

    state = run_pipeline("4.1 Security\nThe supplier must hold ISO 27001.")
    match = state["matches"]["R001"]
    assert match["status"] == "Full"                 # "full" was normalised
    assert match["evidence_ids"] == ["CAP-05"]       # made-up id CAP-999 was dropped
    assert state["answers"]["R001"]["passed"] is True
