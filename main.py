# ========================================
# Streamlit RAG KB + FAISS + Nova Micro LLM
# Chat + Evaluation + LLM-as-Judge
# ========================================

import streamlit as st
import io, pickle, re, time, json, threading
from pathlib import Path
import PyPDF2
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import boto3
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import ndcg_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ================== CONFIG ==================
st.set_page_config(
    page_title="DiaKB — Lab Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
load_dotenv()

AWS_REGION        = "us-east-1"
LOCAL_VECTOR_FILE = "vector_store.pkl"
LOCAL_FAISS_FILE  = "faiss_index.faiss"
LOCAL_EMBED_CACHE = "embed_cache.pkl"
LOCAL_BM25_FILE   = "bm25_index.pkl"   # FIX: persist BM25 so it survives restarts


# ================== MODELS ==================
@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker_model = load_reranker()
EMBED_CACHE: dict = {}

# ================== AWS ==================
session = boto3.Session(region_name=AWS_REGION)
bedrock = session.client("bedrock-runtime")

# ================== EMBED CACHE ==================
def save_embed_cache():
    with open(LOCAL_EMBED_CACHE, "wb") as f:
        pickle.dump(EMBED_CACHE, f)

def load_embed_cache():
    global EMBED_CACHE
    if Path(LOCAL_EMBED_CACHE).exists():
        with open(LOCAL_EMBED_CACHE, "rb") as f:
            EMBED_CACHE.update(pickle.load(f))

# ================== EMBEDDING ==================
def embed_text(text: str) -> list:
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result    = json.loads(response["body"].read())
    embedding = result["embedding"]
    EMBED_CACHE[text] = embedding
    return embedding

# ================== PDF INGESTION ==================
# Common PDF ligature artifacts produced by PyPDF2 (fi, fl, ff, ffi, ffl, etc.)
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
})

def _clean_pdf_text(text: str) -> str:
    """
    Normalise raw PyPDF2 extraction output.

    Problems fixed:
    - Ligature Unicode characters (fi, fl, ff…) that embed as unknown tokens.
    - Soft-hyphen line breaks ("treat-\nment" → "treatment") that split keywords.
    - Excessive whitespace / blank lines that inflate paragraph count.
    - Header/footer repetition is NOT removed here (would need heuristics per
      document); instead we rely on paragraph splitting to isolate them.
    """
    # 1. Ligatures → ASCII equivalents
    text = text.translate(_LIGATURES)
    # 2. Soft-hyphen word-wrap: "treat-\nment" → "treatment"
    text = re.sub(r"-\n\s*", "", text)
    # 3. Plain line-break mid-word (no hyphen): keep single newlines as spaces
    #    but preserve double newlines (paragraph boundaries).
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 4. Collapse runs of spaces/tabs (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 5. Strip leading/trailing whitespace
    return text.strip()

def load_pdf_text(file_obj) -> list:
    reader = PyPDF2.PdfReader(file_obj)
    pages  = []
    for i, p in enumerate(reader.pages):
        raw = p.extract_text()
        if raw:
            cleaned = _clean_pdf_text(raw)
            if cleaned:
                pages.append({"text": cleaned, "page": i + 1})
    return pages


# ── Chunking constants ────────────────────────────────────────────────────────
# CHUNK_TOKEN_LIMIT: target upper bound in *tokens* per chunk.
#   Titan Embed v2 accepts up to 8192 tokens, but smaller chunks (150-200 tokens)
#   embed more precisely — the model has a single vector to capture the whole
#   chunk, so shorter = more focused semantic signal.
CHUNK_TOKEN_LIMIT  = 180   # ~900 characters of normal prose
# OVERLAP_SENTENCES: how many sentences from the tail of chunk N are prepended
#   to chunk N+1.  This preserves cross-boundary context without duplicating
#   large blocks of text the way character-overlap does.
OVERLAP_SENTENCES  = 2

# Rough tokens-per-character ratio for English prose (GPT/Titan tokenisers
# average ~4 chars/token).  We use this to avoid an actual tokeniser dependency.
_CHARS_PER_TOKEN   = 4.0

def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))

# Sentence splitter: split on '.', '!', '?' followed by whitespace or end-of-string.
# We keep the delimiter attached to the preceding sentence so each sentence is a
# complete unit (e.g. "Dr. Smith said so." stays together).
_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> list:
    """Split text into sentences, filtering out empty strings."""
    raw = _SENT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]

def chunk_text(text: str,
               chunk_token_limit: int = CHUNK_TOKEN_LIMIT,
               overlap_sentences: int = OVERLAP_SENTENCES) -> list:
    """
    Paragraph-first, sentence-aware, token-budget chunker.

    Strategy
    --------
    1. Split the page on paragraph boundaries (one or more blank lines).
       Paragraphs are natural topic units in most documents; we never merge
       content across a paragraph break, which prevents unrelated ideas from
       landing in the same embedding vector.

    2. Within each paragraph, split into sentences using a lightweight regex.
       Sentence-complete chunks embed far better than mid-sentence fragments
       because the embedding model sees a coherent semantic unit.

    3. Greedily pack sentences into the current chunk until the token budget
       would be exceeded, then flush and start a new chunk.  Token budget
       (not character count) is the right control knob because that is what
       the embedding model actually processes.

    4. Carry the last `overlap_sentences` sentences of the finished chunk
       into the next chunk.  This preserves cross-boundary context (e.g. a
       pronoun in sentence N+1 that refers to a noun in sentence N) without
       the large duplicate blocks that character-overlap produces.
    """
    chunks = []

    # Step 1: paragraph split — match one or more blank lines
    paragraphs = re.split(r'\n\s*\n', text)

    for para in paragraphs:
        para = re.sub(r'\s+', ' ', para).strip()
        if not para:
            continue

        # Step 2: sentence split
        sentences = _split_sentences(para)
        if not sentences:
            continue

        # Steps 3 & 4: greedy pack with sentence-level overlap
        current_sentences = []
        current_tokens    = 0

        for sent in sentences:
            sent_tokens = _approx_tokens(sent)

            # Edge case: a single sentence exceeds the budget on its own
            # (e.g. a very long table row).  Emit it as its own chunk rather
            # than silently dropping it or creating an undersized overlap.
            if sent_tokens >= chunk_token_limit:
                if current_sentences:
                    chunks.append(' '.join(current_sentences))
                chunks.append(sent)
                current_sentences = []
                current_tokens    = 0
                continue

            if current_tokens + sent_tokens > chunk_token_limit and current_sentences:
                # Flush current chunk
                chunks.append(' '.join(current_sentences))
                # Seed the next chunk with the overlap tail
                current_sentences = current_sentences[-overlap_sentences:]
                current_tokens    = sum(_approx_tokens(s) for s in current_sentences)

            current_sentences.append(sent)
            current_tokens += sent_tokens

        # Flush the last partial chunk for this paragraph
        if current_sentences:
            chunks.append(' '.join(current_sentences))

    return chunks if chunks else ([text.strip()[:800]] if text.strip() else [])


# ================== VECTOR STORE ==================
def build_vector_store_from_uploads(uploaded_files, vectors=None, progress_callback=None):
    vectors = vectors or []
    total   = len(uploaded_files)
    for idx, file_obj in enumerate(uploaded_files):
        pages = load_pdf_text(file_obj)
        for p in pages:
            for chunk in chunk_text(p["text"]):
                if not chunk.strip():
                    continue
                emb = embed_text(chunk)
                vectors.append({"text": chunk, "embedding": emb, "page": p["page"]})
        if progress_callback:
            progress_callback(int((idx + 1) / total * 100))
    corpus    = [re.sub(r"\s+", " ", v["text"].lower()).split() for v in vectors]
    bm25_index = BM25Okapi(corpus)
    return vectors, bm25_index

# ================== FAISS ==================
def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row so that inner-product search == cosine similarity.
    Titan Embed v2 returns unit-norm vectors but float32 casting can introduce
    tiny drift; explicit normalisation guarantees correctness.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero
    return matrix / norms

def build_faiss_index_with_progress(vectors, progress_callback=None):
    """
    Build a FAISS IndexFlatIP (inner-product / cosine) index.

    Why IndexFlatIP instead of IndexFlatL2?
    Titan Embed v2 produces unit-norm embeddings.  For unit-norm vectors,
    cosine similarity = inner product, and higher inner product = more similar.
    IndexFlatL2 gives LOWER distance for MORE similar docs (inverted ordering),
    which is fine for nearest-neighbour search — but IndexFlatIP is the
    semantically correct index for cosine similarity and avoids any ambiguity.
    """
    dim        = len(vectors[0]["embedding"])
    index      = faiss.IndexFlatIP(dim)               # cosine via inner product
    embeddings = np.array([v["embedding"] for v in vectors]).astype("float32")
    embeddings = _l2_normalise(embeddings)            # ensure unit norm
    batch_size = 50
    total      = len(embeddings)
    for i in range(0, total, batch_size):
        index.add(embeddings[i : i + batch_size])
        if progress_callback:
            progress_callback(min((i + batch_size) / total, 1.0))
        time.sleep(0.01)
    return index

# ================== PERSISTENCE ==================
def save_vector_store_and_faiss(vectors, faiss_index, bm25_index):
    pickle.dump(vectors, open(LOCAL_VECTOR_FILE, "wb"))
    faiss.write_index(faiss_index, LOCAL_FAISS_FILE)
    pickle.dump(bm25_index, open(LOCAL_BM25_FILE, "wb"))   # FIX: persist BM25
    save_embed_cache()

def load_vector_store_local():
    """Load vectors, FAISS index, and BM25 index from disk (all three must exist)."""
    if (
        Path(LOCAL_VECTOR_FILE).exists()
        and Path(LOCAL_FAISS_FILE).exists()
        and Path(LOCAL_BM25_FILE).exists()
    ):
        vectors     = pickle.load(open(LOCAL_VECTOR_FILE, "rb"))
        faiss_index = faiss.read_index(LOCAL_FAISS_FILE)
        bm25_index  = pickle.load(open(LOCAL_BM25_FILE, "rb"))  # FIX: restore BM25
        return vectors, faiss_index, bm25_index
    return None, None, None

# ================== RETRIEVAL ==================
def faiss_retrieve(query: str, top_k: int = 10):
    vectors     = st.session_state.vector_store
    faiss_index = st.session_state.faiss_index
    if not vectors or faiss_index is None:
        return []
    # L2-normalise the query vector to match the normalised index embeddings
    # so inner-product search correctly measures cosine similarity.
    q_emb = np.array([embed_text(query)], dtype="float32")
    q_emb = _l2_normalise(q_emb)
    _, indices = faiss_index.search(q_emb, min(top_k, len(vectors)))
    return [vectors[i]["text"] for i in indices[0] if i < len(vectors)]

def bm25_retrieve(query: str, top_k: int = 10):
    bm25    = st.session_state.get("bm25_index")
    vectors = st.session_state.vector_store
    if bm25 is None or not vectors:
        return []
    tokens  = re.sub(r"\s+", " ", query.lower()).split()
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [vectors[i]["text"] for i in top_idx if scores[i] > 0]

def _reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion (RRF).

    Why RRF instead of simple concat-dedup?
    Concat-dedup treats every retrieved document identically regardless of its
    rank in either list.  A document that ranked #1 in FAISS and #1 in BM25 is
    the strongest possible signal of relevance, but under concat-dedup it gets
    no boost over a document that appeared only once at rank #15.

    RRF score = sum(1 / (k + rank_i)) across all lists.
    k=60 is the standard value from the original Cormack et al. (2009) paper —
    it dampens the impact of very high ranks without over-penalising lower ones.
    Documents present in multiple lists naturally accumulate higher scores.
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def hybrid_retrieve(query: str, top_k: int = 15):
    faiss_hits = faiss_retrieve(query, top_k)
    bm25_hits  = bm25_retrieve(query, top_k)
    return _reciprocal_rank_fusion([faiss_hits, bm25_hits])[:top_k * 2]

def rerank(query: str, docs: list, top_k: int = 3):
    if not docs:
        return [], []
    pairs  = [[query, d] for d in docs]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ranked_docs   = [d for d, _ in ranked[:top_k]]
    ranked_scores = [float(s) for _, s in ranked[:top_k]]
    return ranked_docs, ranked_scores

# ================== TOKEN OVERLAP RELEVANCE ==================
def token_overlap_score(doc: str, gold: str) -> float:
    """Token-level F1 overlap between a retrieved chunk and the gold answer."""
    doc_tokens  = set(re.sub(r"\s+", " ", doc.lower()).split())
    gold_tokens = set(re.sub(r"\s+", " ", gold.lower()).split())
    if not doc_tokens or not gold_tokens:
        return 0.0
    intersection = doc_tokens & gold_tokens
    precision = len(intersection) / len(doc_tokens)
    recall    = len(intersection) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ================== LLM CALL ==================
# Maximum number of prior conversation turns to include in each LLM call.
# Older turns are dropped to avoid silent context-window truncation.
MAX_HISTORY_TURNS = 6

def call_llm(context: str, question: str, history: list) -> str:
    """
    Call Nova Micro with retrieved context and conversation history.

    RAG quality decisions
    ---------------------
    temperature=0.0  — Factual QA should be deterministic.  Higher temperature
                       introduces token-level randomness that causes the model to
                       drift from the context and hallucinate.  0.0 = greedy
                       decoding, always picks the highest-probability token.

    No fallback to free-chat — The original prompt said "if not in context, chat
                       like a normal assistant."  This causes the model to
                       hallucinate plausible-sounding but ungrounded answers
                       whenever the retrieved context is weak.  The correct
                       behaviour is to say "not found" so the user knows to
                       rephrase or add more documents.

    History capped at MAX_HISTORY_TURNS — Passing the full unbounded history
                       eventually exceeds Nova Micro's context window.  We keep
                       only the most recent N turns so the model always has room
                       for the context + answer.
    """
    system_prompt = (
        "You are a precise medical/clinical question-answering assistant.\n\n"
        "Rules:\n"
        "1. Answer ONLY using information in the provided CONTEXT block.\n"
        "2. If the answer cannot be found in the CONTEXT, respond with exactly: "
        "\"The answer is not available in the provided documents.\"\n"
        "3. Be concise — 1 to 3 sentences maximum.\n"
        "4. Do not repeat the question.\n"
        "5. Do not add caveats, disclaimers, or extra explanation beyond what the context supports.\n"
        "6. Use clinical/diagnostic phrasing where appropriate."
    )

    # Cap history to avoid context-window overflow
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]
    messages = [
        {"role": t["role"], "content": [{"text": t["content"]}]}
        for t in recent_history
    ]

    if context.strip():
        user_content = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer in 1-3 sentences using only the context above."
        )
    else:
        # No context retrieved — tell the model explicitly so it gives the
        # "not found" response rather than hallucinating.
        user_content = (
            f"CONTEXT: [No relevant documents were retrieved]\n\n"
            f"QUESTION: {question}"
        )

    messages.append({"role": "user", "content": [{"text": user_content}]})

    payload = {
        "system": [{"text": system_prompt}],
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": 1024,
            "temperature": 0.0,   # deterministic — no hallucination drift
            "topP": 1.0,
        },
    }
    response = bedrock.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    try:
        return result["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        return str(result)

# ================== BACKGROUND INGESTION ==================
_ingest_state = {
    "running":     False,
    "progress":    0.0,       # overall 0.0 → 1.0
    "stage":       "",        # human-readable current stage label
    "stage_index": 0,         # which stage we're on (0, 1, 2)
    "stage_total": 3,         # total stages: embed → faiss → bm25+save
    "file_index":  0,         # which file is being embedded right now
    "file_total":  0,
    "chunks_so_far": 0,       # running chunk count during embed
    "status":      "",
    "done":        False,
    "error":       "",
    "chunk_count": 0,
    "doc_count":   0,
}

# Stage weight allocation (must sum to 1.0):
#   50% embedding  (slowest — one Bedrock call per chunk)
#   35% FAISS build
#   15% BM25 + save
_STAGE_WEIGHTS = [0.50, 0.35, 0.15]

def background_ingest(uploaded_files, existing_vectors):
    global _ingest_state
    try:
        n_files = len(uploaded_files)
        _ingest_state.update(
            running=True, done=False, error="",
            progress=0.0, stage="Embedding PDFs…",
            stage_index=0, file_total=n_files, file_index=0, chunks_so_far=0,
        )

        # ── Stage 0: embed ────────────────────────────────────────────────
        # Progress within this stage is per-file (0 → _STAGE_WEIGHTS[0])
        def embed_progress(file_idx, file_pct):
            """file_pct is 0–100 for the current file."""
            base        = _STAGE_WEIGHTS[0] * (file_idx / n_files)
            within_file = _STAGE_WEIGHTS[0] * (1 / n_files) * (file_pct / 100)
            _ingest_state.update(
                progress    = base + within_file,
                file_index  = file_idx + 1,
                stage       = f"Embedding PDF {file_idx + 1}/{n_files}…",
            )

        vectors = list(existing_vectors)
        for idx, file_obj in enumerate(uploaded_files):
            pages = load_pdf_text(file_obj)
            file_chunks = 0
            # Chunk once per page and reuse — avoids calling chunk_text twice
            # (once to estimate, once to embed) which would double the work.
            page_chunks = [(p["page"], chunk)
                           for p in pages
                           for chunk in chunk_text(p["text"])
                           if chunk.strip()]
            total_chunks_estimate = max(len(page_chunks), 1)
            for page_num, chunk in page_chunks:
                emb = embed_text(chunk)
                vectors.append({"text": chunk, "embedding": emb, "page": page_num})
                file_chunks += 1
                _ingest_state["chunks_so_far"] += 1
                embed_progress(idx, min(file_chunks / total_chunks_estimate * 100, 99))
            embed_progress(idx, 100)   # mark file complete

        # ── Stage 1: FAISS ────────────────────────────────────────────────
        stage1_base = _STAGE_WEIGHTS[0]
        _ingest_state.update(stage="Building FAISS index…", stage_index=1,
                             progress=stage1_base)

        def faiss_progress(pct):   # pct is 0.0 → 1.0
            _ingest_state["progress"] = stage1_base + _STAGE_WEIGHTS[1] * pct

        faiss_index = build_faiss_index_with_progress(vectors, progress_callback=faiss_progress)

        # ── Stage 2: BM25 + BM25 index build ─────────────────────────────
        stage2_base = _STAGE_WEIGHTS[0] + _STAGE_WEIGHTS[1]
        _ingest_state.update(stage="Building BM25 index…", stage_index=2,
                             progress=stage2_base)
        corpus     = [re.sub(r"\s+", " ", v["text"].lower()).split() for v in vectors]
        bm25_index = BM25Okapi(corpus)

        _ingest_state.update(stage="Saving to disk…", progress=stage2_base + _STAGE_WEIGHTS[2] * 0.5)
        save_vector_store_and_faiss(vectors, faiss_index, bm25_index)

        _ingest_state["result"] = (vectors, faiss_index, bm25_index)
        _ingest_state.update(
            running=False, done=True, progress=1.0,
            status="✓ Ingestion complete.",
            stage="Done",
            chunk_count=len(vectors),
            doc_count=n_files,
        )
    except Exception as e:
        _ingest_state.update(running=False, done=False,
                             error=str(e), status="❌ Ingestion failed.", stage="")

# ================== SESSION INIT ==================
load_embed_cache()

if _ingest_state.get("done") and "result" in _ingest_state:
    vectors, faiss_index, bm25_index = _ingest_state.pop("result")
    st.session_state.vector_store   = vectors
    st.session_state.faiss_index    = faiss_index
    st.session_state.bm25_index     = bm25_index
    st.session_state.kb_chunk_count = len(vectors)
    _ingest_state["done"]           = False

if "vector_store" not in st.session_state:
    vectors, faiss_index, bm25_index = load_vector_store_local()
    st.session_state.vector_store   = vectors or []
    st.session_state.faiss_index    = faiss_index
    st.session_state.bm25_index     = bm25_index
    st.session_state.kb_chunk_count = len(vectors) if vectors else 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== THEME & CSS ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');

/* ── Global tokens ── */
:root {
    --bg:           #f6f8fa;
    --bg-surface:   #ffffff;
    --bg-raised:    #eaeef2;
    --border:       #d0d7de;
    --border-soft:  #e4e8ec;
    --accent:       #0969da;
    --accent-dim:   #dbeafe;
    --accent-glow:  rgba(9,105,218,.12);
    --green:        #1a7f37;
    --green-dim:    #dcfce7;
    --amber:        #9a6700;
    --amber-dim:    #fef9c3;
    --red:          #cf222e;
    --red-dim:      #fef2f2;
    --text-primary: #1f2328;
    --text-secondary:#57606a;
    --text-muted:   #8c959f;
    --radius:       8px;
    --radius-lg:    12px;
    --font-sans:    'IBM Plex Sans', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
}

/* ── App shell ── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background: var(--bg) !important;
    color: var(--text-primary) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem 6rem !important; max-width: 1100px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 1px 0 0 var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-mono) !important;
    font-size: .75rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    margin: 1.2rem 0 .5rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: .8rem !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: .65rem 1.4rem !important;
    margin-bottom: -1px !important;
    transition: color .15s, border-color .15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
    letter-spacing: .05em !important;
    background: var(--bg-raised) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: .45rem 1.1rem !important;
    transition: border-color .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    color: var(--accent) !important;
}
.stButton > button:disabled {
    opacity: .35 !important;
    cursor: not-allowed !important;
}

/* ── Primary CTA button (ingest) ── */
.btn-primary > button {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, [data-testid="stChatInput"] textarea {
    font-family: var(--font-sans) !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border-top: 1px solid var(--border) !important;
    padding: .75rem 1rem !important;
}
[data-testid="stChatInput"] textarea {
    font-size: .95rem !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: .25rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown p {
    font-size: .95rem !important;
    line-height: 1.65 !important;
}
/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-lg) !important;
    padding: .75rem 1rem !important;
    margin: .3rem 0 !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--accent-dim) !important;
    border: 1px solid rgba(9,105,218,.2) !important;
    border-radius: var(--radius-lg) !important;
    padding: .75rem 1rem !important;
    margin: .3rem 0 !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
}
[data-testid="stSlider"] .stSlider div[data-baseweb="slider"] div {
    background: var(--accent) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), #79c0ff) !important;
    border-radius: 99px !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--bg-raised) !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: .9rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: .78rem !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.5rem !important; font-family: var(--font-mono) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
.dvn-scroller { background: var(--bg-surface) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color .15s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Utility classes ── */
.mono { font-family: var(--font-mono) !important; }
.label {
    font-family: var(--font-mono);
    font-size: .7rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.pill {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .06em;
    padding: .15rem .5rem;
    border-radius: 99px;
    line-height: 1.6;
}
.pill-green  { background: var(--green-dim);  color: var(--green);  border: 1px solid var(--green); }
.pill-amber  { background: var(--amber-dim);  color: var(--amber);  border: 1px solid var(--amber); }
.pill-blue   { background: var(--accent-dim); color: var(--accent); border: 1px solid var(--accent); }
.pill-red    { background: var(--red-dim);    color: var(--red);    border: 1px solid var(--red); }
.pill-muted  { background: var(--bg-raised);  color: var(--text-muted); border: 1px solid var(--border); }

.section-label {
    font-family: var(--font-mono);
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .45rem;
    padding: .18rem .45rem;
    background: var(--accent-dim);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    display: inline-block;
}
.metric-group-label {
    font-family: var(--font-mono);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-dim);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    padding: .2rem .55rem;
    margin: 1.4rem 0 .7rem;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
chunk_count = st.session_state.get("kb_chunk_count", 0)

def _kb_status_html() -> str:
    if _ingest_state["running"]:
        return '<span class="pill pill-blue">⏳ Indexing…</span>'
    elif chunk_count > 0:
        return f'<span class="pill pill-green">✓ {chunk_count:,} chunks</span>'
    else:
        return '<span class="pill pill-amber">Empty</span>'

# ================== SIDEBAR ==================
with st.sidebar:

    # ── Wordmark ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:.6rem 0 1rem">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;
                  color:#1f2328;letter-spacing:-.01em;">MedKB</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:.75rem;
                  color:#8b949e;margin-top:.1rem;">Clinical Knowledge Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Knowledge Base section ────────────────────────────────────────────
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)

    # Status pill
    st.markdown(_kb_status_html(), unsafe_allow_html=True)

    if chunk_count > 0 and not _ingest_state["running"]:
        pages_seen = {v.get("page") for v in st.session_state.vector_store}
        st.markdown(
            f'<div style="font-size:.78rem;color:#57606a;margin:.3rem 0 .6rem;">'
            f'{len(pages_seen)} pages &nbsp;·&nbsp; {chunk_count:,} chunks</div>',
            unsafe_allow_html=True,
        )

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Ingestion progress
    if _ingest_state["running"]:
        stage      = _ingest_state.get("stage", "Working…")
        progress   = _ingest_state.get("progress", 0.0)
        stage_idx  = _ingest_state.get("stage_index", 0)
        chunks_done= _ingest_state.get("chunks_so_far", 0)

        STAGE_NAMES = ["Embed", "FAISS", "BM25"]
        pills_html  = " ".join(
            f'<span class="pill {"pill-blue" if i == stage_idx else "pill-green" if i < stage_idx else "pill-muted"}">'
            f'{"→ " if i == stage_idx else "✓ " if i < stage_idx else ""}{STAGE_NAMES[i]}</span>'
            for i in range(3)
        )
        st.markdown(f'<div style="margin:.5rem 0 .3rem">{pills_html}</div>', unsafe_allow_html=True)
        st.progress(min(progress, 1.0))
        detail = f"{stage}  ·  {chunks_done:,} chunks" if stage_idx == 0 else stage
        st.markdown(f'<div style="font-size:.73rem;color:#57606a;margin-top:.2rem">{detail}</div>',
                    unsafe_allow_html=True)
        time.sleep(0.4)
        st.rerun()
    elif _ingest_state.get("error"):
        st.markdown(
            f'<div class="pill pill-red" style="margin:.4rem 0">⚠ {_ingest_state["error"][:60]}</div>',
            unsafe_allow_html=True,
        )
    elif _ingest_state.get("status") and not _ingest_state["running"]:
        st.markdown(
            f'<div style="font-size:.75rem;color:#1a7f37;margin:.3rem 0">{_ingest_state["status"]}</div>',
            unsafe_allow_html=True,
        )

    if uploaded_files:
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        col_btn, col_clear = st.columns([3, 2])
        with col_btn:
            if st.button("▶ Ingest", disabled=_ingest_state["running"], use_container_width=True):
                threading.Thread(
                    target=background_ingest,
                    args=(uploaded_files, list(st.session_state.vector_store)),
                    daemon=True,
                ).start()
                st.rerun()
        with col_clear:
            if chunk_count > 0 and st.button("Reset KB", use_container_width=True):
                st.session_state.vector_store   = []
                st.session_state.faiss_index    = None
                st.session_state.bm25_index     = None
                st.session_state.kb_chunk_count = 0
                for f in [LOCAL_VECTOR_FILE, LOCAL_FAISS_FILE, LOCAL_BM25_FILE, LOCAL_EMBED_CACHE]:
                    Path(f).unlink(missing_ok=True)
                st.rerun()

    # ── Settings section ──────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.4rem">Retrieval</div>',
                unsafe_allow_html=True)

    top_k_rerank = st.slider(
        "Chunks returned to LLM", min_value=1, max_value=8, value=3,
        help="Number of reranked chunks fed into the answer prompt.",
    )
    show_debug = st.checkbox("Show retrieval debug", value=False)

    # ── Conversation section ───────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Conversation</div>',
                unsafe_allow_html=True)

    turn_count = len(st.session_state.chat_history) // 2
    st.markdown(
        f'<div style="font-size:.78rem;color:#57606a;margin-bottom:.5rem">{turn_count} turn{"s" if turn_count != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Clear history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="position:fixed;bottom:1rem;left:0;width:15rem;text-align:center;
                font-family:'IBM Plex Mono',monospace;font-size:.63rem;color:#8c959f;">
        Nova Micro · Titan Embed v2<br>FAISS · BM25 · RRF
    </div>
    """, unsafe_allow_html=True)

# ================== MAIN TABS ==================

# ── KB build progress banner (shown across both tabs while indexing) ──────────
if _ingest_state["running"]:
    stage      = _ingest_state.get("stage", "Working…")
    progress   = _ingest_state.get("progress", 0.0)
    stage_idx  = _ingest_state.get("stage_index", 0)
    chunks_done= _ingest_state.get("chunks_so_far", 0)
    file_idx   = _ingest_state.get("file_index", 0)
    file_total = _ingest_state.get("file_total", 1)
    pct_int    = int(progress * 100)

    STAGE_NAMES  = ["① Embed", "② FAISS", "③ BM25 + Save"]
    stage_pills  = " &nbsp; ".join(
        (
            '<span style="font-family:IBM Plex Mono,monospace;font-size:.68rem;'
            'font-weight:' + ("700" if i == stage_idx else "400") + ";"
            'color:' + ("#0969da" if i == stage_idx else "#1a7f37" if i < stage_idx else "#8c959f") + ';">'
            + ("▶ " if i == stage_idx else "✓ " if i < stage_idx else "") + STAGE_NAMES[i] + "</span>"
        )
        for i in range(3)
    )
    detail = (
        f"File {file_idx}/{file_total} &nbsp;·&nbsp; {chunks_done:,} chunks embedded"
        if stage_idx == 0 else stage
    )
    st.markdown(
        f"""
        <div style="background:#dbeafe;border:1px solid #93c5fd;border-radius:10px;
                    padding:.75rem 1.1rem .6rem;margin-bottom:1rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;
                      margin-bottom:.45rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;
                        font-weight:700;color:#1e40af;letter-spacing:.04em;">
              BUILDING KNOWLEDGE BASE &nbsp;— &nbsp;{pct_int}%
            </div>
            <div style="font-size:.72rem;color:#3b82f6;">{detail}</div>
          </div>
          <div style="background:#bfdbfe;border-radius:99px;height:7px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,#2563eb,#60a5fa);
                        border-radius:99px;height:100%;width:{pct_int}%;
                        transition:width .3s ease;"></div>
          </div>
          <div style="margin-top:.45rem;">{stage_pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

# ================== CHAT TAB ==================
with tab_chat:

    # ── Page header ───────────────────────────────────────────────────────
    kb_pill = _kb_status_html()
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:.8rem;margin-bottom:1.2rem">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:1.35rem;
                       font-weight:600;color:#1f2328;">Clinical Q&amp;A</span>
          {kb_pill}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Empty state ───────────────────────────────────────────────────────
    history_container = st.container()
    with history_container:
        if not st.session_state.chat_history:
            if chunk_count == 0:
                st.markdown("""
                <div style="margin:3rem auto;max-width:420px;text-align:center;">
                  <div style="font-size:2rem;margin-bottom:.8rem">📄</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:.95rem;
                              font-weight:700;color:#1f2328;margin-bottom:.4rem">
                    No documents indexed
                  </div>
                  <div style="font-size:.85rem;color:#57606a;line-height:1.6">
                    Upload one or more PDFs via the sidebar<br>then click <strong>▶ Ingest</strong> to build the knowledge base.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="margin:3rem auto;max-width:420px;text-align:center;">
                  <div style="font-size:2rem;margin-bottom:.8rem">🩺</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:.95rem;
                              font-weight:700;color:#1f2328;margin-bottom:.4rem">
                    Knowledge base ready
                  </div>
                  <div style="font-size:.85rem;color:#57606a;line-height:1.6">
                    {chunk_count:,} chunks indexed — ask a clinical question below.
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            for turn in st.session_state.chat_history:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    question = st.chat_input(
        "Ask a clinical question…" if chunk_count > 0 else "Index documents first…",
        disabled=_ingest_state["running"],
    )

    if question:
        with history_container:
            with st.chat_message("user"):
                st.markdown(question)

        context, final_docs, rerank_scores = "", [], []
        if st.session_state.vector_store:
            candidates = hybrid_retrieve(question)
            final_docs, rerank_scores = rerank(question, candidates, top_k=top_k_rerank)
            context = "\n\n".join(final_docs)

        history_for_llm = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.chat_history
        ]

        with history_container:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving & generating…"):
                    answer = call_llm(context, question, history_for_llm)
                st.markdown(answer)

                # Source attribution strip
                if final_docs:
                    pages = []
                    vs = st.session_state.vector_store
                    for doc in final_docs:
                        for v in vs:
                            if v["text"] == doc:
                                pages.append(v.get("page", "?"))
                                break
                    unique_pages = sorted(set(pages))
                    page_str = ", ".join(f"p.{p}" for p in unique_pages)
                    st.markdown(
                        f'<div style="margin-top:.5rem;font-size:.73rem;'
                        f'color:#57606a;font-family:\'IBM Plex Mono\',monospace;">'
                        f'Sources: {page_str}</div>',
                        unsafe_allow_html=True,
                    )

                # Retrieval debug
                if show_debug:
                    with st.expander("Retrieval debug", expanded=False):
                        dcols = st.columns(3)
                        dcols[0].caption("FAISS top-5")
                        for h in faiss_retrieve(question, 5):
                            dcols[0].markdown(
                                f'<div style="font-size:.72rem;color:#57606a;'
                                f'border:1px solid #d0d7de;border-radius:6px;'
                                f'padding:.4rem .6rem;margin:.2rem 0;'
                                f'font-family:\'IBM Plex Mono\',monospace;">{h[:120]}…</div>',
                                unsafe_allow_html=True,
                            )
                        dcols[1].caption("BM25 top-5")
                        for h in bm25_retrieve(question, 5):
                            dcols[1].markdown(
                                f'<div style="font-size:.72rem;color:#57606a;'
                                f'border:1px solid #d0d7de;border-radius:6px;'
                                f'padding:.4rem .6rem;margin:.2rem 0;'
                                f'font-family:\'IBM Plex Mono\',monospace;">{h[:120]}…</div>',
                                unsafe_allow_html=True,
                            )
                        dcols[2].caption(f"Reranked (top {top_k_rerank})")
                        for doc, score in zip(final_docs, rerank_scores):
                            dcols[2].markdown(
                                f'<div style="font-size:.72rem;color:#57606a;'
                                f'border:1px solid #d0d7de;border-radius:6px;'
                                f'padding:.4rem .6rem;margin:.2rem 0;'
                                f'font-family:\'IBM Plex Mono\',monospace;">'
                                f'<span style="color:#0969da">{score:+.3f}</span> &nbsp;{doc[:100]}…</div>',
                                unsafe_allow_html=True,
                            )

        st.session_state.chat_history.append({"role": "user",      "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ================== LLM AS JUDGE ==================
def llm_judge_score(llm_answer: str, gold_answer: str, question: str) -> dict:
    judge_prompt = f"""You are an evaluation assistant.
Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {llm_answer}

Score the Generated Answer on four dimensions (0–5 each):
- correctness: Is the answer factually correct compared to the gold answer?
- relevance: Does the answer address the question?
- completeness: Does the answer cover all key points in the gold answer?
- similarity: How similar is the answer to the gold answer in content and style?

Respond ONLY with a valid JSON object, no extra text:
{{"correctness": <int>, "relevance": <int>, "completeness": <int>, "similarity": <int>}}"""

    payload = {
        "system": [{"text": "You are a strict evaluation assistant. Output only valid JSON."}],
        "messages": [{"role": "user", "content": [{"text": judge_prompt}]}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0.0, "topP": 1.0},
    }
    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        result   = json.loads(response["body"].read())
        raw_text = result["output"]["message"]["content"][0]["text"].strip()
        raw_text = re.sub(r"```json|```", "", raw_text).strip()
        scores   = json.loads(raw_text)
        return {k: max(0, min(5, int(scores[k]))) for k in ["correctness", "relevance", "completeness", "similarity"]}
    except Exception as e:
        st.warning(f"Judge scoring failed: {e}")
        return {"correctness": 0, "relevance": 0, "completeness": 0, "similarity": 0}

# ================== EVALUATION TAB ==================
with tab_eval:

    st.markdown("""
    <div style="margin-bottom:1.4rem">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.35rem;
                  font-weight:600;color:#1f2328;">Evaluation</div>
      <div style="font-size:.83rem;color:#57606a;margin-top:.25rem">
        Upload a labelled CSV/JSON with <code style="font-family:'IBM Plex Mono',monospace;
        color:#0969da;background:#1f3a5c;padding:.05rem .3rem;border-radius:4px">question</code>
        and <code style="font-family:'IBM Plex Mono',monospace;color:#0969da;
        background:#1f3a5c;padding:.05rem .3rem;border-radius:4px">answer</code> columns.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_eval_file = st.file_uploader(
        "Upload evaluation file", type=["csv", "json"], key="eval_upload",
        label_visibility="collapsed",
    )

    if uploaded_eval_file:
        eval_data = (
            pd.read_json(uploaded_eval_file)
            if uploaded_eval_file.name.endswith(".json")
            else pd.read_csv(uploaded_eval_file)
        )

        # Preview card
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:.5rem;'
            f'background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;'
            f'padding:.5rem .9rem;margin:.4rem 0 1rem;font-size:.8rem;color:#8b949e;">'
            f'<span style="color:#3fb950">✓</span>'
            f'<span style="font-family:\'IBM Plex Mono\',monospace">{uploaded_eval_file.name}</span>'
            f'<span style="color:#8c959f">·</span>'
            f'<span>{len(eval_data)} rows</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("▶ Run Evaluation", type="primary"):
            results  = []
            rouge_sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            errors   = []
            smoothie = SmoothingFunction().method1
            total_rows = len(eval_data)

            # Progress area
            prog_label = st.empty()
            prog_bar   = st.progress(0)
            prog_label.markdown(
                '<div style="font-size:.78rem;color:#57606a;font-family:\'IBM Plex Mono\','
                'monospace">Evaluating row 0 / ' + str(total_rows) + '…</div>',
                unsafe_allow_html=True,
            )

            for idx, row in eval_data.iterrows():
                try:
                    query = str(row["question"])
                    gold  = str(row["answer"])

                    candidates            = hybrid_retrieve(query)
                    final_docs, rerank_sc = rerank(query, candidates, top_k=top_k_rerank)
                    context               = "\n\n".join(final_docs)
                    llm_answer            = call_llm(context, query, [])

                    overlap_scores  = [token_overlap_score(d, gold) for d in final_docs]
                    relevance_flags = [1 if s >= 0.1 else 0 for s in overlap_scores]
                    k = len(final_docs)

                    precision_at_k = sum(relevance_flags) / k if k > 0 else 0.0
                    recall_at_k    = min(sum(relevance_flags), 1)

                    n = min(len(relevance_flags), len(rerank_sc))
                    if n > 0 and sum(relevance_flags[:n]) > 0:
                        ideal    = sorted(relevance_flags[:n], reverse=True)
                        pred     = rerank_sc[:n]
                        min_pred = min(pred)
                        if min_pred < 0:
                            pred = [s - min_pred for s in pred]
                        ndcg = ndcg_score([ideal], [pred])
                    else:
                        ndcg = 0.0

                    hyp_tokens = llm_answer.split()
                    ref_tokens = gold.split()
                    bleu = sentence_bleu(
                        [ref_tokens], hyp_tokens,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie,
                    ) if hyp_tokens and ref_tokens else 0.0

                    rs     = rouge_sc.score(gold, llm_answer)
                    rouge1 = rs["rouge1"].fmeasure
                    rougeL = rs["rougeL"].fmeasure

                    judge      = llm_judge_score(llm_answer, gold, query)
                    judge_norm = {k: v / 5.0 for k, v in judge.items()}

                    results.append({
                        "question":              query,
                        "gold_answer":           gold,
                        "llm_answer":            llm_answer,
                        "precision@K":           round(precision_at_k, 4),
                        "recall@K":              round(recall_at_k, 4),
                        "nDCG@K":                round(ndcg, 4),
                        "BLEU":                  round(bleu, 4),
                        "ROUGE-1":               round(rouge1, 4),
                        "ROUGE-L":               round(rougeL, 4),
                        "judge_correctness":     judge["correctness"],
                        "judge_relevance":       judge["relevance"],
                        "judge_completeness":    judge["completeness"],
                        "judge_similarity":      judge["similarity"],
                        "judge_correctness_norm":  round(judge_norm["correctness"], 3),
                        "judge_relevance_norm":    round(judge_norm["relevance"], 3),
                        "judge_completeness_norm": round(judge_norm["completeness"], 3),
                        "judge_similarity_norm":   round(judge_norm["similarity"], 3),
                    })

                except Exception as e:
                    errors.append({"row": idx, "question": row.get("question", ""), "error": str(e)})

                pct = (idx + 1) / total_rows
                prog_bar.progress(pct)
                prog_label.markdown(
                    f'<div style="font-size:.78rem;color:#57606a;font-family:\'IBM Plex Mono\','
                    f'monospace">Evaluating row {idx + 1} / {total_rows}…</div>',
                    unsafe_allow_html=True,
                )

            prog_label.empty()
            prog_bar.empty()

            if results:
                df = pd.DataFrame(results)

                # ── Summary banner ─────────────────────────────────────────
                n_ok  = len(results)
                n_err = len(errors)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:.6rem;'
                    f'background:#dcfce7;border:1px solid #1a7f37;border-radius:8px;'
                    f'padding:.65rem 1rem;margin:.5rem 0 1.4rem;font-size:.82rem;">'
                    f'<span style="color:#1a7f37;font-size:1rem">✓</span>'
                    f'<span style="color:#1f2328"><strong>{n_ok}</strong> rows evaluated</span>'
                    + (f'<span style="color:#8c959f">·</span>'
                       f'<span style="color:#cf222e"><strong>{n_err}</strong> failed</span>' if n_err else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )

                # ── Metric cards ───────────────────────────────────────────
                st.markdown('<div class="metric-group-label">Retrieval</div>', unsafe_allow_html=True)
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Precision@K", f"{df['precision@K'].mean():.3f}")
                rc2.metric("Recall@K",    f"{df['recall@K'].mean():.3f}")
                rc3.metric("nDCG@K",      f"{df['nDCG@K'].mean():.3f}")

                st.markdown('<div class="metric-group-label">Generation</div>', unsafe_allow_html=True)
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric("BLEU",    f"{df['BLEU'].mean():.3f}")
                gc2.metric("ROUGE-1", f"{df['ROUGE-1'].mean():.3f}")
                gc3.metric("ROUGE-L", f"{df['ROUGE-L'].mean():.3f}")

                st.markdown('<div class="metric-group-label">LLM Judge &nbsp;<span style="font-size:.65rem;color:#8c959f">(0–1, higher is better)</span></div>',
                            unsafe_allow_html=True)
                jc1, jc2, jc3, jc4 = st.columns(4)
                jc1.metric("Correctness",  f"{df['judge_correctness_norm'].mean():.3f}")
                jc2.metric("Relevance",    f"{df['judge_relevance_norm'].mean():.3f}")
                jc3.metric("Completeness", f"{df['judge_completeness_norm'].mean():.3f}")
                jc4.metric("Similarity",   f"{df['judge_similarity_norm'].mean():.3f}")

                # ── Per-row table ──────────────────────────────────────────
                st.markdown('<div class="metric-group-label" style="margin-top:1.6rem">Per-question results</div>',
                            unsafe_allow_html=True)
                display_cols = ["question", "llm_answer", "precision@K", "recall@K",
                                "nDCG@K", "BLEU", "ROUGE-1", "ROUGE-L",
                                "judge_correctness_norm", "judge_relevance_norm"]
                st.dataframe(df[display_cols], use_container_width=True, height=320)

                # ── Download ───────────────────────────────────────────────
                st.download_button(
                    "⬇  Download full results CSV",
                    data=df.to_csv(index=False).encode(),
                    file_name="eval_results.csv",
                    mime="text/csv",
                )

            if errors:
                with st.expander(f"⚠  {len(errors)} rows failed"):
                    st.dataframe(pd.DataFrame(errors))