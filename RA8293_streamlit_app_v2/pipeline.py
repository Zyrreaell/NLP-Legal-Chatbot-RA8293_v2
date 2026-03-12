from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cohere
import faiss
import nltk
import numpy as np
import pandas as pd
from bert_score import BERTScorer
from groq import Groq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# PATHS / CONSTANTS

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

BM25_PATH = DATA_DIR / "bm25.pkl"
FAISS_PATH = DATA_DIR / "faiss_legal.index"
CHUNKS_PATH = DATA_DIR / "chunks_for_rag.jsonl"

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
COHERE_CHAT_MODEL = "command-r-08-2024"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

DEFAULT_BM25_K = 60
DEFAULT_SBERT_K = 60
DEFAULT_HYBRID_TOP_K = 30
DEFAULT_FINAL_TOP_DOCS = 8
DEFAULT_RRF_K = 60

MODELS_TO_COMPARE = [
    "cohere/command-r-08-2024",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

MODEL_DISPLAY_NAMES = {
    "cohere/command-r-08-2024": "Cohere Command R",
    "openai/gpt-oss-120b": "GPT-OSS 120B (Groq)",
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Groq)",
}

# Groq Qwen judge
JUDGE_MODEL = "qwen/qwen3-32b"

# Legal-BERTScore model
LEGAL_BERTSCORE_MODEL = "nlpaueb/legal-bert-base-uncased"
LEGAL_BERTSCORE_NUM_LAYERS = 12

LEGAL_SYSTEM_PROMPT = (
    "You are a highly precise legal assistant specializing in the Intellectual Property "
    "Code of the Philippines (RA 8293). Your task is to answer the user's question "
    "based STRICTLY on the provided documents.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. STRICT GROUNDING: Answer ONLY using the provided text. If the text does not "
    "contain the answer, say exactly: 'The retrieved text does not contain enough "
    "information to answer this.'\n"
    "2. EXHAUSTIVE PRECISION: You must include all relevant legal conditions, specific "
    "timeframes (e.g., months/years), and exceptions mentioned in the text. Do not "
    "oversimplify.\n"
    "3. COMPARATIVE LOGIC: If a question asks for a difference or comparison between "
    "two concepts, you must explicitly state the contrasting elements of BOTH subjects.\n"
    "4. CAREFUL READING: Pay strict attention to legal qualifiers (e.g., 'fixed' vs. "
    "'unfixed', 'exclusive' vs. 'non-exclusive'). Do not conflate distinct legal categories.\n"
    "5. CONCISENESS: Be direct, objective, and strictly factual."
)

# NLTK SETUP


def _ensure_nltk() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


_ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))

# SECRET HELPERS


def _get_streamlit_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except Exception:
        pass
    return None


def set_api_keys(
    cohere_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    cohere_api_key = (
        cohere_api_key
        or os.environ.get("COHERE_API_KEY")
        or _get_streamlit_secret("COHERE_API_KEY")
    )
    groq_api_key = (
        groq_api_key
        or os.environ.get("GROQ_API_KEY")
        or _get_streamlit_secret("GROQ_API_KEY")
    )

    if not cohere_api_key:
        raise RuntimeError(
            "COHERE_API_KEY was not found. Put it in .streamlit/secrets.toml locally "
            "or Streamlit Cloud secrets."
        )

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Put it in .streamlit/secrets.toml locally "
            "or Streamlit Cloud secrets."
        )

    os.environ["COHERE_API_KEY"] = cohere_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key

    return {"cohere": cohere_api_key, "groq": groq_api_key}


@lru_cache(maxsize=1)
def get_cohere_client() -> cohere.ClientV2:
    keys = set_api_keys()
    return cohere.ClientV2(api_key=keys["cohere"])


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    keys = set_api_keys()
    return Groq(api_key=keys["groq"])


# FILE / RESOURCE LOADERS


def _ensure_required_files() -> None:
    missing = [str(p) for p in [BM25_PATH, FAISS_PATH, CHUNKS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required runtime files:\n- " + "\n- ".join(missing)
        )


@lru_cache(maxsize=1)
def load_chunks() -> pd.DataFrame:
    _ensure_required_files()

    rows: List[Dict[str, Any]] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("chunks_for_rag.jsonl was loaded, but it is empty.")

    if "chunk_id" not in df.columns:
        df["chunk_id"] = [f"chunk_{i}" for i in range(len(df))]

    if "citation" not in df.columns:
        df["citation"] = df["chunk_id"]

    if "text" not in df.columns:
        raise ValueError("chunks_for_rag.jsonl must contain a 'text' field.")

    df["chunk_id"] = df["chunk_id"].astype(str)
    df["citation"] = df["citation"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_bm25() -> BM25Okapi:
    _ensure_required_files()
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_faiss_index():
    _ensure_required_files()
    return faiss.read_index(str(FAISS_PATH))


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

@lru_cache(maxsize=1)
def get_legal_bertscorer() -> BERTScorer:
    return BERTScorer(
        model_type=LEGAL_BERTSCORE_MODEL,
        num_layers=LEGAL_BERTSCORE_NUM_LAYERS,
        lang="en",
        device="cpu",
        rescale_with_baseline=False,
    )

# TEXT / RESPONSE HELPERS


def bm25_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^\w\.]+", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _cohere_message_to_text(response: Any) -> str:
    try:
        content = response.message.content
        if isinstance(content, list) and len(content) > 0:
            first = content[0]
            if hasattr(first, "text"):
                return str(first.text).strip()
            if isinstance(first, dict) and "text" in first:
                return str(first["text"]).strip()
        return str(content).strip()
    except Exception:
        return ""


def _groq_message_to_text(response: Any) -> str:
    try:
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def unique_citations(top_df: pd.DataFrame) -> List[str]:
    if top_df is None or top_df.empty:
        return []
    seen = set()
    citations = []
    for cite in top_df["citation"].astype(str).tolist():
        if cite not in seen:
            citations.append(cite)
            seen.add(cite)
    return citations


def format_context_string(top_df: pd.DataFrame) -> str:
    if top_df is None or top_df.empty:
        return ""

    blocks = []
    for _, row in top_df.iterrows():
        citation = str(row.get("citation", row.get("chunk_id", "Unknown")))
        text = str(row.get("text", ""))
        blocks.append(f"[{citation}]\n{text}")

    return "\n\n".join(blocks)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        try:
            raw = re.sub(r",\s*}", "}", raw)
            raw = re.sub(r",\s*]", "]", raw)
            return json.loads(raw)
        except Exception:
            return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# RETRIEVAL


def bm25_retrieve(query: str, top_k: int = 20) -> pd.DataFrame:
    df = load_chunks()
    bm25 = load_bm25()

    q_tokens = bm25_tokenize(query)
    scores = np.asarray(bm25.get_scores(q_tokens), dtype=float)

    if scores.size == 0:
        return pd.DataFrame(columns=["chunk_id", "citation", "text", "bm25_score"])

    top_k = min(top_k, len(scores))
    top_idx = np.argsort(scores)[::-1][:top_k]

    out = df.iloc[top_idx].copy()
    out["bm25_score"] = scores[top_idx]
    return out[["chunk_id", "citation", "text", "bm25_score"]].reset_index(drop=True)


def sbert_retrieve(query: str, top_k: int = 20) -> pd.DataFrame:
    df = load_chunks()
    index = load_faiss_index()
    embed_model = load_embedder()

    safe_top_k = min(top_k, index.ntotal)
    if safe_top_k <= 0:
        return pd.DataFrame(columns=["chunk_id", "citation", "text", "sbert_score"])

    q = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, idxs = index.search(q, safe_top_k)

    idxs_1d = idxs[0]
    scores_1d = scores[0]

    valid_mask = idxs_1d >= 0
    idxs_1d = idxs_1d[valid_mask]
    scores_1d = scores_1d[valid_mask]

    if len(idxs_1d) == 0:
        return pd.DataFrame(columns=["chunk_id", "citation", "text", "sbert_score"])

    out = df.iloc[idxs_1d].copy()
    out["sbert_score"] = scores_1d
    return out[["chunk_id", "citation", "text", "sbert_score"]].reset_index(drop=True)


def hybrid_retrieve(
    query: str,
    bm25_k: int = DEFAULT_BM25_K,
    sbert_k: int = DEFAULT_SBERT_K,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    k_param: int = DEFAULT_RRF_K,
) -> pd.DataFrame:
    b = bm25_retrieve(query, top_k=bm25_k).copy()
    s = sbert_retrieve(query, top_k=sbert_k).copy()

    if b.empty and s.empty:
        return pd.DataFrame(columns=["chunk_id", "citation", "text", "hybrid_score"])

    if not b.empty:
        b["bm25_rank"] = range(1, len(b) + 1)
    else:
        b = pd.DataFrame(columns=["chunk_id", "citation", "text", "bm25_rank"])

    if not s.empty:
        s["sbert_rank"] = range(1, len(s) + 1)
    else:
        s = pd.DataFrame(columns=["chunk_id", "citation", "text", "sbert_rank"])

    merged = pd.merge(
        b,
        s,
        on=["chunk_id", "citation", "text"],
        how="outer",
    )

    merged["bm25_rank"] = merged["bm25_rank"].fillna(float("inf"))
    merged["sbert_rank"] = merged["sbert_rank"].fillna(float("inf"))

    def compute_rrf(row: pd.Series) -> float:
        score = 0.0
        if row["bm25_rank"] != float("inf"):
            score += 1.0 / (k_param + row["bm25_rank"])
        if row["sbert_rank"] != float("inf"):
            score += 1.0 / (k_param + row["sbert_rank"])
        return score

    merged["hybrid_score"] = merged.apply(compute_rrf, axis=1)
    merged = (
        merged.sort_values("hybrid_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    return merged[["chunk_id", "citation", "text", "hybrid_score"]]


# COHERE RERANK + DOCUMENT PACKAGING


def build_cohere_documents(
    top_df: pd.DataFrame,
    max_chars: int = 2000,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    top_df = top_df.copy()

    def to_safe_id(raw: str, idx: int) -> str:
        raw = str(raw)
        base = re.sub(r"\s+", "_", raw.strip())
        base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
        if not base:
            base = f"chunk_{idx}"
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
        return f"{base}_{h}"

    top_df["cohere_id"] = [
        to_safe_id(v, i) for i, v in enumerate(top_df["chunk_id"].astype(str))
    ]

    documents: List[Dict[str, Any]] = []
    for _, row in top_df.iterrows():
        citation = str(row.get("citation", row["chunk_id"]))
        snippet = str(row["text"])[:max_chars]

        documents.append(
            {
                "id": row["cohere_id"],
                "data": {
                    "title": citation,
                    "snippet": snippet,
                },
            }
        )

    return documents, top_df


def rerank_with_cohere(
    query: str,
    candidates_df: pd.DataFrame,
    top_k: int = DEFAULT_FINAL_TOP_DOCS,
) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(
            columns=["chunk_id", "citation", "text", "hybrid_score", "rerank_score"]
        )

    co = get_cohere_client()
    docs = candidates_df["text"].fillna("").astype(str).tolist()
    safe_top_k = min(top_k, len(docs))

    results = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=safe_top_k,
    )

    top_indices = [res.index for res in results.results]

    out = candidates_df.iloc[top_indices].copy()
    out["rerank_score"] = [res.relevance_score for res in results.results]

    return out.reset_index(drop=True)


# QUERY REWRITING


def rewrite_query(query: str, history: Optional[Sequence[Any]] = None) -> str:
    if not history:
        return query

    co = get_cohere_client()

    conversation_lines = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, ai_msg = item[0], item[1]
        elif isinstance(item, dict):
            user_msg = item.get("user", "")
            ai_msg = item.get("assistant", "")
        else:
            continue

        conversation_lines.append(f"User: {user_msg}")
        conversation_lines.append(f"AI: {ai_msg}")

    conv = "\n".join(conversation_lines).strip()
    if not conv:
        return query

    prompt = (
        "Given the following chat history, rewrite the user's latest query into a "
        "single, standalone search query that can be used to accurately search a "
        "legal database. Do not answer the question, just provide the rewritten query.\n\n"
        f"History:\n{conv}\n\n"
        f"Latest Query: {query}\n\n"
        "Rewritten Search Query:"
    )

    response = co.chat(
        model=COHERE_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    rewritten = _cohere_message_to_text(response)
    return rewritten if rewritten else query


# GENERATION


def generate_cohere_native(
    query: str,
    documents: List[Dict[str, Any]],
    id_to_cite: Dict[str, str],
) -> Tuple[str, float, List[str]]:
    start_time = time.time()

    try:
        co = get_cohere_client()

        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response = co.chat(
            model=COHERE_CHAT_MODEL,
            messages=messages,
            documents=documents,
            temperature=0.1,
            max_tokens=600,
        )

        latency = time.time() - start_time
        answer = _cohere_message_to_text(response)

        used_cites: List[str] = []
        try:
            citations_obj = getattr(response.message, "citations", None)
            if citations_obj:
                used_ids = set()
                for citation in citations_obj:
                    sources = getattr(citation, "sources", []) or []
                    for src in sources:
                        src_id = getattr(src, "id", None)
                        if src_id:
                            used_ids.add(str(src_id))

                for doc_id in used_ids:
                    used_cites.append(id_to_cite.get(doc_id, doc_id))
        except Exception:
            pass

        return answer, round(latency, 2), used_cites

    except Exception as e:
        latency = time.time() - start_time
        return f"API Error: {str(e)}", round(latency, 2), []


def generate_groq_model(
    model: str,
    query: str,
    context_str: str,
) -> Tuple[str, float]:
    start_time = time.time()
    try:
        client = get_groq_client()

        system_prompt = (
            f"{LEGAL_SYSTEM_PROMPT}\n\n"
            "Use only the context below.\n\n"
            f"CONTEXT:\n{context_str}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=600,
        )

        latency = time.time() - start_time
        answer = _groq_message_to_text(response)
        return answer, round(latency, 2)

    except Exception as e:
        latency = time.time() - start_time
        return f"API Error: {str(e)}", round(latency, 2)


# HIGH-LEVEL PIPELINE FUNCTIONS


def retrieve_context_for_query(
    query: str,
    history: Optional[Sequence[Any]] = None,
    rewrite: bool = False,
    bm25_k: int = DEFAULT_BM25_K,
    sbert_k: int = DEFAULT_SBERT_K,
    hybrid_top_k: int = DEFAULT_HYBRID_TOP_K,
    top_docs: int = DEFAULT_FINAL_TOP_DOCS,
) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query is empty.")

    effective_query = rewrite_query(query, history) if rewrite else query

    candidates = hybrid_retrieve(
        effective_query,
        bm25_k=bm25_k,
        sbert_k=sbert_k,
        top_k=hybrid_top_k,
    )

    top_df = rerank_with_cohere(effective_query, candidates, top_k=top_docs)
    context_str = format_context_string(top_df)
    context_list = top_df["text"].tolist() if not top_df.empty else []

    documents, top_df = build_cohere_documents(top_df)

    id_to_cite = dict(
        zip(
            top_df["cohere_id"].astype(str),
            top_df["citation"].astype(str),
        )
    )

    return {
        "original_query": query,
        "effective_query": effective_query,
        "candidates_df": candidates,
        "top_df": top_df,
        "context_list": context_list,
        "context_str": context_str,
        "documents": documents,
        "id_to_cite": id_to_cite,
        "sources": unique_citations(top_df),
    }


def run_comparative_rag(
    query: str,
    history: Optional[Sequence[Any]] = None,
    rewrite: bool = False,
    top_docs: int = DEFAULT_FINAL_TOP_DOCS,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not query or not query.strip():
        return {
            "effective_query": "",
            "sources": [],
            "top_chunks": [],
            "answers": {},
            "latencies": {},
            "cited_sources": [],
            "context_str": "",
            "context_list": [],
        }

    models = models or MODELS_TO_COMPARE

    retrieved = retrieve_context_for_query(
        query=query,
        history=history,
        rewrite=rewrite,
        top_docs=top_docs,
    )

    answers: Dict[str, str] = {}
    latencies: Dict[str, float] = {}

    cohere_answer, cohere_latency, cohere_cites = generate_cohere_native(
        query=query,
        documents=retrieved["documents"],
        id_to_cite=retrieved["id_to_cite"],
    )

    answers["cohere/command-r-08-2024"] = cohere_answer
    latencies["cohere/command-r-08-2024"] = cohere_latency

    for model in models:
        if model == "cohere/command-r-08-2024":
            continue

        answer, latency = generate_groq_model(
            model=model,
            query=query,
            context_str=retrieved["context_str"],
        )
        answers[model] = answer
        latencies[model] = latency

    return {
        "effective_query": retrieved["effective_query"],
        "sources": retrieved["sources"],
        "cited_sources": cohere_cites,
        "top_chunks": retrieved["top_df"].to_dict(orient="records"),
        "context_str": retrieved["context_str"],
        "context_list": retrieved["context_list"],
        "answers": answers,
        "latencies": latencies,
    }


# EVALUATION HELPERS


def pairwise_legal_bertscore(
    answers: Dict[str, str],
) -> Tuple[Dict[str, float], Optional[str]]:
    usable = [(m, (a or "").strip()) for m, a in answers.items() if (a or "").strip()]
    out = {m: 0.0 for m in answers.keys()}

    if len(usable) < 2:
        return out, "Need at least 2 non-empty answers."

    accum: Dict[str, List[float]] = {m: [] for m, _ in usable}

    try:
        scorer = get_legal_bertscorer()

        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                m1, a1 = usable[i]
                m2, a2 = usable[j]

                _, _, f1 = scorer.score([a1], [a2])
                s = float(f1[0].item())

                accum[m1].append(s)
                accum[m2].append(s)

        for model, vals in accum.items():
            out[model] = round(float(np.mean(vals)) if vals else 0.0, 4)

        return out, None

    except Exception as e:
        return out, repr(e)

def _groq_json_judge(
    system_prompt: str,
    user_prompt: str,
    model: str = JUDGE_MODEL,
    max_tokens: int = 220,
) -> Optional[dict]:
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_completion_tokens=max_tokens,
        reasoning_effort="none",
        reasoning_format="hidden",
        response_format={"type": "json_object"},
    )

    raw = _groq_message_to_text(response)
    try:
        return json.loads(raw)
    except Exception:
        return None


def live_nli_judge(query: str, context: str, answer: str) -> Tuple[float, str]:
    if not context.strip() or not answer.strip():
        return 0.0, "NoData"

    system_prompt = "Return only JSON."

    user_prompt = f"""
Evaluate whether the answer is grounded in the context.

Return JSON only:
{{
  "score": 0.0,
  "verdict": "Entailment"
}}

Allowed verdicts:
- Entailment
- Neutral
- Contradiction

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()

    try:
        data = _groq_json_judge(system_prompt, user_prompt, model=JUDGE_MODEL)
        if not data:
            return 0.0, "ParsingError"

        score = _clamp01(float(data["score"]))
        verdict = str(data["verdict"]).strip()

        if verdict not in {"Entailment", "Neutral", "Contradiction"}:
            return 0.0, "ParsingError"

        return round(score, 4), verdict
    except Exception as e:
        return 0.0, f"JudgeError: {e}"


def live_ragas_eval(
    query: str,
    context_list: List[str],
    answer: str,
) -> Tuple[float, float]:
    if not context_list or not answer.strip():
        return 0.0, 0.0

    context = "\n\n".join(context_list[:5])

    system_prompt = "Return only JSON."

    user_prompt = f"""
Evaluate the answer against the question and context.

Return JSON only:
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0
}}

Both values must be between 0 and 1.

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()

    try:
        data = _groq_json_judge(system_prompt, user_prompt, model=JUDGE_MODEL)
        if not data:
            return 0.0, 0.0

        faithfulness = _clamp01(float(data["faithfulness"]))
        answer_relevancy = _clamp01(float(data["answer_relevancy"]))

        return round(faithfulness, 4), round(answer_relevancy, 4)
    except Exception:
        return 0.0, 0.0


def build_live_eval_tables(
    query: str,
    result: Dict[str, Any],
    run_eval: bool = False,
) -> Dict[str, Any]:
    answers = result.get("answers", {})
    latencies = result.get("latencies", {})
    context_list = result.get("context_list", []) or []
    sources = result.get("sources", []) or []
    cited_sources = result.get("cited_sources", []) or []

    primary_context = "\n\n".join(context_list[:3]) if context_list else ""

    latency_rows = []
    for model in MODELS_TO_COMPARE:
        latency_rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES.get(model, model),
                "Latency (s)": float(latencies.get(model, 0.0)),
            }
        )
    latency_df = pd.DataFrame(latency_rows)

    bertscore_df = pd.DataFrame(columns=["Model", "Legal-BERTScore F1"])
    nli_df = pd.DataFrame(columns=["Model", "Score", "Verdict"])
    ragas_df = pd.DataFrame(columns=["Model", "Faithfulness", "Answer Relevancy"])

    errors: Dict[str, str] = {}

    if run_eval:
        try:
            bert_scores, bert_err = pairwise_legal_bertscore(answers)
            bert_rows = []
            for model in MODELS_TO_COMPARE:
                bert_rows.append(
                    {
                        "Model": MODEL_DISPLAY_NAMES.get(model, model),
                        "Legal-BERTScore F1": float(bert_scores.get(model, 0.0)),
                    }
                )
            bertscore_df = pd.DataFrame(bert_rows)

            if bert_err:
                errors["bertscore"] = bert_err

        except Exception as e:
            errors["bertscore"] = str(e)

        try:
            nli_rows = []
            for model in MODELS_TO_COMPARE:
                score, verdict = live_nli_judge(
                    query=query,
                    context=primary_context,
                    answer=answers.get(model, ""),
                )
                nli_rows.append(
                    {
                        "Model": MODEL_DISPLAY_NAMES.get(model, model),
                        "Score": float(score),
                        "Verdict": verdict,
                    }
                )
            nli_df = pd.DataFrame(nli_rows)
        except Exception as e:
            errors["nli"] = str(e)

        try:
            ragas_rows = []
            for model in MODELS_TO_COMPARE:
                faithfulness, answer_relevancy = live_ragas_eval(
                    query=query,
                    context_list=context_list,
                    answer=answers.get(model, ""),
                )
                ragas_rows.append(
                    {
                        "Model": MODEL_DISPLAY_NAMES.get(model, model),
                        "Faithfulness": float(faithfulness),
                        "Answer Relevancy": float(answer_relevancy),
                    }
                )
            ragas_df = pd.DataFrame(ragas_rows)
        except Exception as e:
            errors["ragas"] = str(e)


    return {
        "latency_df": latency_df,
        "bertscore_df": bertscore_df,
        "nli_df": nli_df,
        "ragas_df": ragas_df,
        "sources": sources,
        "cited_sources": cited_sources,
        "errors": errors,
        "run_eval": run_eval,
    }

# UTILITIES FOR STREAMLIT APP

def warmup(
    cohere_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> None:
    set_api_keys(cohere_api_key=cohere_api_key, groq_api_key=groq_api_key)
    load_chunks()
    load_bm25()
    load_faiss_index()
    load_embedder()
    get_cohere_client()
    get_groq_client()


def healthcheck() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "root": str(ROOT),
        "data_dir_exists": DATA_DIR.exists(),
        "bm25_exists": BM25_PATH.exists(),
        "faiss_exists": FAISS_PATH.exists(),
        "chunks_exists": CHUNKS_PATH.exists(),
        "judge_model": JUDGE_MODEL,
        "compare_models": MODELS_TO_COMPARE,
    }

    try:
        df = load_chunks()
        info["num_chunks"] = int(len(df))
        info["chunk_columns"] = list(df.columns)
    except Exception as e:
        info["chunks_error"] = str(e)

    try:
        index = load_faiss_index()
        info["faiss_ntotal"] = int(index.ntotal)
    except Exception as e:
        info["faiss_error"] = str(e)

    try:
        set_api_keys()
        info["cohere_key_loaded"] = bool(os.environ.get("COHERE_API_KEY"))
        info["groq_key_loaded"] = bool(os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        info["secrets_error"] = str(e)

    return info


if __name__ == "__main__":
    warmup()
    print("Warmup complete.")
    print(healthcheck())

    sample_query = "What are the remedies for infringement under RA 8293?"
    result = run_comparative_rag(sample_query)
    eval_tables = build_live_eval_tables(sample_query, result, run_eval=True)

    print("\nQUERY:", sample_query)
    print("\nANSWERS:")
    for k, v in result["answers"].items():
        print(f"\n--- {k} ---\n{v[:500]}")

    print("\nLATENCY TABLE:")
    print(eval_tables["latency_df"])