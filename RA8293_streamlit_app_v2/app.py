from __future__ import annotations

import traceback
from typing import Dict, List

import streamlit as st

from pipeline import (
    MODEL_DISPLAY_NAMES,
    MODELS_TO_COMPARE,
    build_live_eval_tables,
    healthcheck,
    run_comparative_rag,
    warmup,
)

# PAGE CONFIG

st.set_page_config(
    page_title="RA 8293 Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "example_query" not in st.session_state:
    st.session_state.example_query = ""

if "eval_tables" not in st.session_state:
    st.session_state.eval_tables = None

if "last_run_eval" not in st.session_state:
    st.session_state.last_run_eval = False

# STARTUP / WARMUP

startup_error = None
try:
    warmup()
except Exception as e:
    startup_error = str(e)

# HELPERS

def reset_app() -> None:
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.last_query = ""
    st.session_state.example_query = ""
    st.session_state.eval_tables = None
    st.session_state.last_run_eval = False


def render_model_card(title: str, answer: str, latency: float) -> None:
    st.subheader(title)
    st.caption(f"Latency: {latency:.2f}s")
    st.write(answer if answer else "No output returned.")


def render_sources(sources: List[str], cited_sources: List[str]) -> None:
    st.subheader("Sources")

    if cited_sources:
        st.markdown("**Cohere cited:**")
        for src in cited_sources:
            st.markdown(f"- `{src}`")

    if sources:
        st.markdown("**Top retrieved sources:**")
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.info("No sources were returned.")


def render_top_chunks(top_chunks: List[Dict]) -> None:
    st.subheader("Top Retrieved Chunks")

    if not top_chunks:
        st.info("No chunks available.")
        return

    for i, chunk in enumerate(top_chunks, start=1):
        citation = chunk.get("citation", f"Chunk {i}")
        chunk_id = chunk.get("chunk_id", "N/A")
        rerank_score = chunk.get("rerank_score", None)
        hybrid_score = chunk.get("hybrid_score", None)
        text = chunk.get("text", "")

        with st.expander(f"{i}. {citation}"):
            st.markdown(f"**Chunk ID:** `{chunk_id}`")
            if rerank_score is not None:
                st.markdown(f"**Rerank Score:** `{float(rerank_score):.6f}`")
            if hybrid_score is not None:
                st.markdown(f"**Hybrid Score:** `{float(hybrid_score):.6f}`")
            st.markdown("**Text:**")
            st.write(text)


def render_history() -> None:
    if not st.session_state.history:
        return

    with st.expander("Conversation History", expanded=False):
        for i, turn in enumerate(st.session_state.history, start=1):
            st.markdown(f"**Turn {i} — User**")
            st.write(turn.get("user", ""))
            st.markdown(f"**Turn {i} — Assistant (Cohere)**")
            st.write(turn.get("assistant", ""))
            st.markdown("---")

# SIDEBAR

with st.sidebar:
    st.title("Settings")

    rewrite_query = st.checkbox(
        "Rewrite follow-up query using chat history",
        value=True,
        help="Useful for follow-up questions like 'What about the exceptions?'",
    )

    top_docs = st.slider(
        "Final reranked documents",
        min_value=3,
        max_value=12,
        value=8,
        step=1,
        help="How many documents are passed to the answering models after reranking.",
    )

    run_eval = st.checkbox(
        "Run live evaluation metrics",
        value=False,
        help="Runs Legal-BERTScore, Qwen NLI, and Qwen RAGAS-style judging for all 3 models.",
    )

    st.markdown("---")
    st.markdown("### Models in comparison mode")
    for model_name in MODELS_TO_COMPARE:
        st.markdown(f"- {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")

    st.markdown("---")
    if st.button("Clear conversation and results", use_container_width=True):
        reset_app()
        st.rerun()

    st.markdown("---")
    with st.expander("System status", expanded=False):
        try:
            info = healthcheck()
            st.json(info)
        except Exception as e:
            st.error(f"Healthcheck failed: {e}")

# MAIN HEADER

st.title("⚖️ RA 8293 Legal Assistant")
st.caption(
    "Comparison mode: Cohere Command R vs GPT-OSS 120B (Groq) vs Llama 3.3 70B (Groq)"
)

if startup_error:
    st.error(
        "Startup failed. This usually means a missing secret or missing data file.\n\n"
        f"Details: {startup_error}"
    )
    st.stop()

# QUICK EXAMPLES

st.markdown("### Example questions")
example_cols = st.columns(3)

example_1 = "What are the remedies for infringement under RA 8293?"
example_2 = "What is the difference between copyright and related rights under RA 8293?"
example_3 = "How long does copyright protection last for literary works?"

with example_cols[0]:
    if st.button("Example 1", use_container_width=True):
        st.session_state.example_query = example_1

with example_cols[1]:
    if st.button("Example 2", use_container_width=True):
        st.session_state.example_query = example_2

with example_cols[2]:
    if st.button("Example 3", use_container_width=True):
        st.session_state.example_query = example_3

# QUERY INPUT

default_query = st.session_state.example_query or st.session_state.last_query

with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Ask a legal question about RA 8293",
        value=default_query,
        height=120,
        placeholder="Enter your question here...",
    )
    submitted = st.form_submit_button("Run 3-model comparison", use_container_width=True)

# RUN PIPELINE

if submitted:
    clean_query = query.strip()

    if not clean_query:
        st.warning("Please enter a question first.")
    else:
        st.session_state.last_query = clean_query
        st.session_state.example_query = ""

        try:
            with st.spinner("Running retrieval, reranking, comparison, and evaluation..."):
                result = run_comparative_rag(
                    query=clean_query,
                    history=st.session_state.history,
                    rewrite=rewrite_query,
                    top_docs=top_docs,
                    models=MODELS_TO_COMPARE,
                )

                eval_tables = build_live_eval_tables(
                    query=clean_query,
                    result=result,
                    run_eval=run_eval,
                )

            st.session_state.last_result = result
            st.session_state.eval_tables = eval_tables
            st.session_state.last_run_eval = run_eval

            cohere_answer = result.get("answers", {}).get("cohere/command-r-08-2024", "")
            st.session_state.history.append(
                {
                    "user": clean_query,
                    "assistant": cohere_answer,
                }
            )

        except Exception as e:
            st.error(f"An error occurred while running the pipeline: {e}")
            with st.expander("Full error details"):
                st.code(traceback.format_exc(), language="python")

# RESULTS

result = st.session_state.last_result
eval_tables = st.session_state.eval_tables

if result:
    st.markdown("---")
    st.subheader("Effective Query Used for Retrieval")
    st.write(result.get("effective_query", st.session_state.last_query))

    answers = result.get("answers", {})
    latencies = result.get("latencies", {})

    col1, col2, col3 = st.columns(3)

    model_order = [
        "cohere/command-r-08-2024",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
    ]

    columns = [col1, col2, col3]

    for col, model_name in zip(columns, model_order):
        with col:
            render_model_card(
                title=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                answer=answers.get(model_name, "No answer returned."),
                latency=float(latencies.get(model_name, 0.0)),
            )

    if eval_tables:
        st.markdown("---")
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.subheader("Live Evaluation Metrics")

            st.markdown("#### Latency")
            st.dataframe(
                eval_tables["latency_df"],
                use_container_width=True,
                hide_index=True,
            )

            if st.session_state.last_run_eval:
                st.markdown("#### Legal-BERTScore F1 (Pairwise Agreement)")
                st.dataframe(
                    eval_tables["bertscore_df"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### NLI Grounding Check (Qwen on Groq)")
                st.dataframe(
                    eval_tables["nli_df"],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### RAGAS (Faithfulness & Relevancy)")
                st.dataframe(
                    eval_tables["ragas_df"],
                    use_container_width=True,
                    hide_index=True,
                )

                if eval_tables.get("errors"):
                    st.warning("Some evaluation blocks failed.")
                    st.json(eval_tables["errors"])
            else:
                st.info(
                    "Enable 'Run live evaluation metrics' in the sidebar and resubmit "
                    "to show BERTScore, NLI, and RAGAS."
                )

        with right_col:
            st.subheader("Retrieved Sources")

            st.markdown("**Retrieved Sources (shared across all models):**")
            for src in eval_tables.get("sources", []):
                st.markdown(f"- `{src}`")

            cited_sources = eval_tables.get("cited_sources", [])
            if cited_sources:
                st.markdown("**Cohere cited:**")
                for src in cited_sources:
                    st.markdown(f"- `{src}`")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(
        ["Sources", "Top Retrieved Chunks", "Conversation History"]
    )

    with tab1:
        render_sources(
            sources=result.get("sources", []),
            cited_sources=result.get("cited_sources", []),
        )

    with tab2:
        render_top_chunks(result.get("top_chunks", []))

    with tab3:
        render_history()

else:
    st.info("Submit a question to compare the 3 models.")