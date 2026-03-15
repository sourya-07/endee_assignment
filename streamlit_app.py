import logging
import os
import sys
import tempfile

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    APP_TITLE,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    OLLAMA_MODEL,
)
from src.rag_chain import RAGChain
from src.subjects_db import load_subjects, create_or_get_subject, get_subject_index_name
from src.ingest import ingest_documents

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="Multi-Subject Notes",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Neobrutalist CSS
NEOBRUTALIST_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800;900&display=swap" rel="stylesheet">

<style>
/* ── Variables ── */
:root {
    --bg:       #F0EDE5;
    --accent:   #004643;
    --paper:    #ffffff;
    --yellow:   #fdf577;
    --yellow2:  #fce83a;
    --border:   3px solid var(--accent);
    --shadow:   5px 5px 0px var(--accent);
    --shadow-sm:2px 2px 0px var(--accent);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--accent) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-weight: 900 !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px !important;
}

/* ── App header box ── */
.neo-header {
    border: var(--border);
    box-shadow: var(--shadow);
    background: var(--bg);
    padding: 24px 32px;
    margin-bottom: 24px;
}
.neo-header h1 { font-size: 2rem; margin: 0 0 6px 0; }
.neo-header p  { font-weight: 600; margin: 0; }
.status-chip {
    display: inline-block;
    background: var(--yellow);
    border: var(--border);
    box-shadow: var(--shadow-sm);
    padding: 5px 14px;
    font-weight: 800;
    font-size: 0.82rem;
    text-transform: uppercase;
    margin-top: 10px;
}

/* ── Cards / containers ── */
.neo-card {
    border: var(--border);
    box-shadow: var(--shadow);
    background: var(--paper);
    padding: 20px 24px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.78rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--accent);
    margin-bottom: 8px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: var(--border) !important;
    gap: 0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    border: var(--border) !important;
    border-bottom: none !important;
    background: var(--bg) !important;
    color: var(--accent) !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 10px 20px !important;
    margin-right: -3px !important;
}
.stTabs [aria-selected="true"][data-baseweb="tab"] {
    background: var(--paper) !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    border: var(--border) !important;
    box-shadow: var(--shadow) !important;
    background: var(--yellow) !important;
    color: var(--accent) !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 10px 24px !important;
    transition: all .1s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--yellow2) !important;
    box-shadow: var(--shadow-sm) !important;
    transform: translate(3px, 3px) !important;
}
.stButton > button:active {
    transform: translate(5px, 5px) !important;
    box-shadow: 0 0 0 var(--accent) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    border: var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    background: var(--paper) !important;
    color: var(--accent) !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    box-shadow: 1px 1px 0 var(--accent) !important;
    transform: translate(2px, 2px);
    outline: none !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    border: var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    background: var(--paper) !important;
    color: var(--accent) !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    border: var(--border) !important;
    background: var(--paper) !important;
    border-radius: 0 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [data-testid="stThumbValue"] {
    background: var(--accent) !important;
    color: var(--yellow) !important;
    border: var(--border) !important;
    border-radius: 0 !important;
    font-weight: 800 !important;
}

/* ── Checkbox ── */
.stCheckbox label { font-weight: 700 !important; color: var(--accent) !important; }
input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stTextInput label, .stTextArea label {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    letter-spacing: .05em !important;
    color: var(--accent) !important;
}

/* ── Dataframe / Tables ── */
.stDataFrame table  { border: var(--border) !important; }
.stDataFrame thead tr th {
    background: var(--yellow) !important;
    color: var(--accent) !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    border-bottom: var(--border) !important;
}

/* ── Info / success / warning boxes ── */
.stAlert {
    border: var(--border) !important;
    border-radius: 0 !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    border: var(--border) !important;
    border-radius: 0 !important;
    background: var(--bg) !important;
    color: var(--accent) !important;
}
</style>
"""

# Helpers

def _llm_status() -> str:
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})" if OPENAI_API_KEY else "Warning: OpenAI — no API key set"
    elif provider == "google":
        return f"Google Gemini ({os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash')})" if GOOGLE_API_KEY else "Warning: Google Gemini — no API key set"
    elif provider == "ollama":
        return f"Ollama ({OLLAMA_MODEL})"
    return f"Unknown: {LLM_PROVIDER}"


def _format_sources(sources: list) -> str:
    if not sources:
        return "No sources retrieved."
    lines = []
    for i, s in enumerate(sources, 1):
        score = s.get("score", 0.0)
        title = s.get("title", "Unknown")
        url = s.get("url", "")
        cat = s.get("category", "")
        link = f"[{title}]({url})" if url else title
        lines.append(f"**{i}.** {link}  \n→ *Category:* `{cat}` | *Relevance:* `{score:.3f}`")
    return "\n\n".join(lines)


def _format_metrics(faithfulness: float, relevancy: float, latency: float) -> str:
    fp = int(faithfulness * 100)
    rp = int(relevancy * 100)
    return (
        f"| Metric | Score | Visual |\n"
        f"|---|---|---|\n"
        f"| **Faithfulness** | `{faithfulness:.2f}` | {'█' * (fp // 10)}{'░' * (10 - fp // 10)} {fp}% |\n"
        f"| **Answer Relevancy** | `{relevancy:.2f}` | {'█' * (rp // 10)}{'░' * (10 - rp // 10)} {rp}% |\n"
        f"| **Latency** | `{latency:.2f}s` | — |\n"
    )


def get_subject_choices() -> list[str]:
    return list(load_subjects().keys())


# RAG chain singleton
@st.cache_resource(show_spinner="Initialising RAG chain…")
def _get_rag_chain() -> RAGChain:
    return RAGChain(use_reranker=True)


# UI

st.markdown(NEOBRUTALIST_CSS, unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="neo-header">
    <h1>Multi-Subject Notes</h1>
    <p>Organize, ingest, and query your knowledge base — each subject is isolated with its own vector index.</p>
    <div class="status-chip">{_llm_status()}</div>
</div>
""", unsafe_allow_html=True)

# Top-level tabs
ask_tab, manage_tab = st.tabs(["Ask Your Notes", "Manage Subjects"])

# ASK TAB
with ask_tab:
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown('<div class="neo-card">', unsafe_allow_html=True)

        # Subject selector
        st.markdown('<div class="section-title">Subject</div>', unsafe_allow_html=True)
        subjects = get_subject_choices()
        subject = st.selectbox("Subject", subjects, label_visibility="collapsed", key="ask_subject")

        # Question
        st.markdown('<div class="section-title">Your Question</div>', unsafe_allow_html=True)
        question = st.text_area("Ask anything about your notes…", height=120, label_visibility="collapsed", key="question_input")

        # Example buttons
        with st.expander("Try an example"):
            examples = [
                "Summarize the main points of my notes.",
                "What are the key concepts covered?",
                "Explain the most important topics in detail.",
                "What connections exist between different ideas?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex[:20]}"):
                    st.session_state["question_input"] = ex
                    st.rerun()

        # Advanced options
        with st.expander("Advanced Options"):
            top_k = st.slider("Number of chunks to retrieve", 1, 20, 5, key="top_k")
            category_filter = st.selectbox("Category filter", ["All", "lecture", "textbook", "note", "article"], key="cat_filter")
            use_reranker = st.checkbox("Enable cross-encoder reranking", value=True, key="reranker")

        ask_clicked = st.button("Ask", key="ask_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        res_tabs = st.tabs(["Answer", "Sources", "Metrics", "Context"])

        if ask_clicked:
            if not subject:
                with res_tabs[0]:
                    st.warning("Please select a subject first.")
            elif not question or not question.strip():
                with res_tabs[0]:
                    st.warning("Please enter a question.")
            else:
                index_name = get_subject_index_name(subject)
                if not index_name:
                    with res_tabs[0]:
                        st.error(f"Internal error: Index for subject '{subject}' not found.")
                else:
                    with st.spinner("Thinking…"):
                        try:
                            chain = _get_rag_chain()
                            chain.retriever.use_reranker = use_reranker
                            cat = category_filter if category_filter != "All" else None
                            response = chain.run(
                                index_name=index_name,
                                question=question.strip(),
                                top_k=top_k,
                                category_filter=cat,
                                compute_metrics=True,
                            )
                            st.session_state["last_response"] = response
                        except Exception as e:
                            logger.exception("Query failed")
                            st.session_state["last_error"] = str(e)

        resp = st.session_state.get("last_response")
        err = st.session_state.get("last_error")

        with res_tabs[0]:
            if err:
                st.error(f"**Error:** {err}\n\nCheck that Endee is running.")
            elif resp:
                st.markdown(resp.answer)
            else:
                st.markdown("*Your answer will appear here after submitting a question.*")

        with res_tabs[1]:
            if resp:
                st.markdown(_format_sources(resp.sources))
            else:
                st.markdown("*Sources will appear here.*")

        with res_tabs[2]:
            if resp:
                st.markdown(_format_metrics(resp.faithfulness, resp.answer_relevancy, resp.latency_seconds))
            else:
                st.markdown("*Metrics will appear here.*")

        with res_tabs[3]:
            if resp:
                ctx = resp.context or ""
                st.code(ctx[:3000])
            else:
                st.markdown("*Context will appear here.*")


# MANAGE TAB
with manage_tab:
    col1, col2 = st.columns(2, gap="large")

    # Create subject
    with col1:
        st.markdown('<div class="neo-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Create New Subject</div>', unsafe_allow_html=True)
        new_subject_name = st.text_input("Subject Name", placeholder="e.g. Quantum Physics, History, Biology", key="new_subject")
        if st.button("Create Subject", key="create_btn"):
            if not new_subject_name or not new_subject_name.strip():
                st.warning("Please enter a subject name.")
            else:
                create_or_get_subject(new_subject_name.strip())
                st.success(f"Subject **'{new_subject_name.strip()}'** created successfully!")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Add content
    with col2:
        st.markdown('<div class="neo-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Add Content to Subject</div>', unsafe_allow_html=True)

        all_subjects = get_subject_choices()
        ingest_subject = st.selectbox("Target Subject", all_subjects, key="ingest_subject")
        uploaded_files = st.file_uploader("Upload Documents (PDF or .txt)", accept_multiple_files=True, type=["pdf", "txt"], key="uploader")
        links_text = st.text_area("Paste URLs (one per line)", placeholder="https://example.com/article", key="links_text")
        reset_index = st.checkbox("Reset index before ingesting", value=False, key="reset_index")

        if st.button("Process & Add Content", key="ingest_btn"):
            if not ingest_subject:
                st.warning("Please select a subject first.")
            else:
                index_name = get_subject_index_name(ingest_subject)
                if not index_name:
                    st.error(f"Internal error: Index for '{ingest_subject}' not found.")
                else:
                    docs = []
                    if uploaded_files:
                        for f in uploaded_files:
                            try:
                                if f.name.lower().endswith(".pdf"):
                                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                        tmp.write(f.read())
                                        tmp_path = tmp.name
                                    with open(tmp_path, "rb") as pf:
                                        reader = PyPDF2.PdfReader(pf)
                                        content = "\n".join(
                                            p.extract_text() or "" for p in reader.pages
                                        )
                                    os.unlink(tmp_path)
                                else:
                                    content = f.read().decode("utf-8", errors="ignore")
                                docs.append({"title": f.name, "content": content})
                            except Exception as e:
                                st.error(f"Error reading {f.name}: {e}")

                    links = [l.strip() for l in (links_text or "").splitlines() if l.strip()]

                    if not docs and not links and not reset_index:
                        st.warning("No documents, links, or reset requested.")
                    else:
                        with st.spinner("Processing…"):
                            try:
                                count = ingest_documents(index_name=index_name, docs=docs, links=links, reset=reset_index)
                                st.success(f"Ingestion complete for **'{ingest_subject}'**! **{count}** vectors indexed.")
                            except Exception as e:
                                logger.exception("Ingest failed")
                                st.error(f"Ingestion failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
