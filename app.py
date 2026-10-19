"""
Web app for the RFP response team (built with Streamlit).

Start it from the project folder with:
    streamlit run app.py

Pick a sample RFP or upload your own, run the agents, review and edit the
answers, approve them, then download the compliance matrix and draft response.
"""

import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from rfp_agent import config, llm
from rfp_agent.documents import read_rfp
from rfp_agent.graph import run_pipeline
from rfp_agent.outputs import STATUS_LABELS, build_rows, export_excel, export_word, pretty_name

st.set_page_config(page_title="RFP Response Agent", layout="wide")

SAMPLES = {
    "Sample: Harbourview Logistics (AI customer service platform)": config.SAMPLE_RFP_DIR / "harbourview_logistics_rfp.pdf",
    "Sample: Aldermoor University (AI student help assistant)": config.SAMPLE_RFP_DIR / "aldermoor_university_rfp.pdf",
}

# ---------- Sidebar: what the agents do ----------
with st.sidebar:
    st.header("The agent team")
    st.markdown(
        "**1. Reader** pulls every requirement out of the RFP.\n\n"
        "**2. Matcher** searches the capability library and decides Full, Partial or Not met.\n\n"
        "**3. Writer** drafts each answer using only the evidence found.\n\n"
        "**4. Reviewer** scores each draft and sends weak ones back to the Writer.\n\n"
        "**You** approve the answers before anything is exported."
    )
    st.divider()
    st.caption(f"Bidding company: {config.COMPANY_NAME}")
    st.caption(f"Model provider: {config.LLM_PROVIDER}")
    if config.LLM_PROVIDER == "demo":
        st.info("Demo mode uses simple rules, not a real model. Set LLM_PROVIDER in .env for real results.")

st.title("RFP Response Agent")
st.write("Turn an RFP into a compliance matrix and a reviewed first-draft response.")

# ---------- Step 1: choose the RFP ----------
choice = st.radio("Which RFP?", list(SAMPLES) + ["Upload my own"])
uploaded = st.file_uploader("RFP file", type=["pdf", "docx", "txt"]) if choice == "Upload my own" else None

if st.button("Run the agents", type="primary"):
    if choice == "Upload my own":
        if uploaded is None:
            st.error("Please upload a file first.")
            st.stop()
        # read_rfp needs a file on disk, so save the upload to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
        rfp_path, rfp_name = Path(tmp.name), Path(uploaded.name).stem
    else:
        rfp_path, rfp_name = SAMPLES[choice], SAMPLES[choice].stem

    llm.reset_stats()
    with st.status("Agents at work...", expanded=True) as status:
        state = run_pipeline(read_rfp(rfp_path), on_progress=st.write)
        status.update(label="Agents finished", state="complete")

    st.session_state["rows"] = build_rows(state)
    st.session_state["log"] = state["log"]
    st.session_state["rfp_name"] = rfp_name
    st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

# ---------- Step 2: results ----------
if "rows" in st.session_state:
    rows = st.session_state["rows"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requirements", len(rows))
    c2.metric("Fully met", sum(r["status"] == "Full" for r in rows))
    c3.metric("Partly met", sum(r["status"] == "Partial" for r in rows))
    c4.metric("Not met", sum(r["status"] == "None" for r in rows))
    c5.metric("Answers rewritten", sum(r["drafts"] > 1 for r in rows))

    tab_review, tab_gaps, tab_trace = st.tabs(["Review and approve", "Needs a human decision", "Agent trace"])

    with tab_review:
        st.write("Edit any answer directly in the table. Answers the Reviewer passed for fully met "
                 "requirements start ticked. Everything else needs your tick before you can export.")
        table = pd.DataFrame([{
            "Approved": r["status"] == "Full" and not r["human_action"],
            "ID": r["id"],
            "Requirement": r["requirement"],
            "Priority": r["priority"],
            "Compliance": STATUS_LABELS[r["status"]],
            "Answer": r["answer"],
            "Score": r["review_score"],
            "Drafts": r["drafts"],
        } for r in rows if r["answer"]])

        edited = st.data_editor(
            table,
            key=f"editor_{st.session_state['run_id']}",
            hide_index=True,
            width="stretch",
            disabled=["ID", "Requirement", "Priority", "Compliance", "Score", "Drafts"],
            column_config={
                "Answer": st.column_config.TextColumn(width="large"),
                "Requirement": st.column_config.TextColumn(width="medium"),
            },
        )

        # Put the human edits back into the rows before exporting.
        edited_answers = dict(zip(edited["ID"], edited["Answer"]))
        final_rows = [{**r, "answer": edited_answers.get(r["id"], r["answer"])} for r in rows]

        pending = int((~edited["Approved"]).sum())
        if pending:
            st.warning(f"{pending} answers still need your approval before export.")
        else:
            st.success("All answers approved. Ready to export.")

        excel_buffer, word_buffer = BytesIO(), BytesIO()
        export_excel(final_rows, excel_buffer)
        export_word(final_rows, pretty_name(st.session_state["rfp_name"]), word_buffer)

        d1, d2 = st.columns(2)
        d1.download_button("Download compliance matrix (Excel)", excel_buffer.getvalue(),
                           file_name=f"{st.session_state['rfp_name']}_compliance_matrix.xlsx",
                           disabled=pending > 0, width="stretch")
        d2.download_button("Download draft response (Word)", word_buffer.getvalue(),
                           file_name=f"{st.session_state['rfp_name']}_draft_response.docx",
                           disabled=pending > 0, width="stretch")

    with tab_gaps:
        gaps = [r for r in rows if r["human_action"]]
        if gaps:
            st.dataframe(pd.DataFrame([{
                "ID": r["id"], "Requirement": r["requirement"], "Priority": r["priority"],
                "Compliance": STATUS_LABELS[r["status"]], "Why": r["reason"], "Action": r["human_action"],
            } for r in gaps]), hide_index=True, width="stretch")
        else:
            st.write("Nothing needs a decision.")

    with tab_trace:
        for message in st.session_state["log"]:
            st.write(message)
        st.divider()
        for r in rows:
            with st.expander(f"{r['id']}  [{r['status']}]  {r['requirement'][:90]}"):
                st.write(f"**Evidence:** {r['evidence'] or 'none found'}")
                st.write(f"**Matcher's reason:** {r['reason']}")
                st.write(f"**Reviewer score:** {r['review_score']}  (drafts: {r['drafts']})")
