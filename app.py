import logging
import os
import sys

import gradio as gr

from src.config import (
    APP_TITLE,
    APP_HOST,
    APP_PORT,
    DEBUG,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    OLLAMA_MODEL,
)
from src.rag_chain import RAGChain
from src.subjects_db import load_subjects, create_or_get_subject, get_subject_index_name
from src.ingest import ingest_documents

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global RAG chain instance (lazy-loaded on first query)
# ─────────────────────────────────────────────────────────────────────────────
_rag_chain: RAGChain | None = None

def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        logger.info("Initialising RAG chain...")
        _rag_chain = RAGChain(use_reranker=True)
    return _rag_chain

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Format sources and metrics
# ─────────────────────────────────────────────────────────────────────────────

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
    faith_bar = "🟩" * int(faithfulness * 10) + "⬜" * (10 - int(faithfulness * 10))
    rel_bar = "🟦" * int(relevancy * 10) + "⬜" * (10 - int(relevancy * 10))
    return (
        f"**Faithfulness** (grounded in context): {faith_bar} `{faithfulness:.2f}`  \n"
        f"**Answer Relevancy** (on-topic): {rel_bar} `{relevancy:.2f}`  \n"
        f"**Latency:** `{latency:.2f}s`"
    )

def _llm_status() -> str:
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        if OPENAI_API_KEY:
            return f"✅ OpenAI (`{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`)"
        return "⚠️ OpenAI — no `OPENAI_API_KEY` set"
    elif provider == "google":
        if GOOGLE_API_KEY:
            return f"✅ Google Gemini (`{os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash')}`)"
        return "⚠️ Google Gemini — no `GOOGLE_API_KEY` set"
    elif provider == "ollama":
        return f"🦙 Ollama (`{OLLAMA_MODEL}`) — make sure Ollama is running"
    return f"❓ Unknown provider: `{LLM_PROVIDER}`"

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_subject_choices():
    return list(load_subjects().keys())

def create_subject(name):
    if not name or not name.strip():
        return "⚠️ Please enter a subject name.", gr.update()
    
    name = name.strip()
    create_or_get_subject(name)
    choices = get_subject_choices()
    
    return f"✅ Subject '{name}' created successfully!", gr.update(choices=choices, value=name)

def ingest_subject_docs(subject, files, links_text, reset):
    if not subject:
        return "⚠️ Please select a subject first."
        
    index_name = get_subject_index_name(subject)
    if not index_name:
        return f"❌ Internal error: Index for subject '{subject}' not found."
        
    docs = []
    if files:
        for f in files:
            try:
                with open(f.name, "r", encoding="utf-8") as file:
                    content = file.read()
                    docs.append({"title": os.path.basename(f.name), "content": content})
            except Exception as e:
                logger.error(f"Error reading file {f.name}: {e}")
                
    links = [l.strip() for l in (links_text or "").split("\n") if l.strip()]
    
    if not docs and not links and not reset:
        return "⚠️ No documents, links, or reset requested."
        
    try:
        count = ingest_documents(index_name=index_name, docs=docs, links=links, reset=reset)
        return f"✅ Ingestion complete for '{subject}'! {count} vectors indexed."
    except Exception as e:
        logger.exception("Ingest failed")
        return f"❌ Ingestion failed: {str(e)}"

def query_rag(subject, question, top_k, category_filter, use_reranker):
    if not subject:
        return "⚠️ Please select a subject first.", "", "", ""
        
    if not question or not question.strip():
        return "⚠️ Please enter a question.", "", "", ""

    index_name = get_subject_index_name(subject)
    if not index_name:
         return f"❌ Internal error: Index for subject '{subject}' not found.", "", "", ""

    try:
        chain = _get_rag_chain()
        chain.retriever.use_reranker = use_reranker

        cat = category_filter if category_filter and category_filter != "All" else None

        response = chain.run(
            index_name=index_name,
            question=question.strip(),
            top_k=top_k,
            category_filter=cat,
            compute_metrics=True,
        )

        answer_md = f"## 🤖 Answer\n\n{response.answer}"
        sources_md = f"## 📚 Sources\n\n{_format_sources(response.sources)}"
        metrics_md = f"## 📊 Eval Metrics\n\n{_format_metrics(response.faithfulness, response.answer_relevancy, response.latency_seconds)}"
        context_md = (
            f"## 🔍 Retrieved Context\n\n```\n{response.context[:3000]}\n```"
            if response.context
            else "*No context retrieved.*"
        )

        return answer_md, sources_md, metrics_md, context_md

    except Exception as e:
        logger.exception("Error in RAG query")
        return (
            f"❌ **Error:** {str(e)}\n\nCheck that Endee is running.",
            "",
            "",
            "",
        )

def refresh_dropdowns():
    choices = get_subject_choices()
    val = choices[0] if choices else None
    return gr.update(choices=choices, value=val), gr.update(choices=choices, value=val)

# ─────────────────────────────────────────────────────────────────────────────
# UI Construction
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
* { box-sizing: border-box; }
body, .gradio-container {
    background-color: #d8d8d8 !important;
    background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(%23noise)" opacity="0.08"/></svg>') !important;
    font-family: 'Helvetica Neue', Arial, sans-serif !important;
    color: #111 !important;
}
/* ── Header ── */
.header-box {
    background: transparent;
    border: 3px solid #111;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: left;
    box-shadow: 6px 6px 0px #111;
}
.header-box h1 {
    font-size: 3rem;
    font-weight: 900;
    color: #111 !important;
    margin: 0 0 8px 0;
    text-transform: lowercase;
    letter-spacing: -2px;
    background: none !important;
    -webkit-text-fill-color: #111 !important;
}
.header-box p { color: #111; font-size: 1.1rem; margin: 0; font-weight: bold; text-transform: lowercase; }
/* ── Cards & Panels ── */
.card, .gr-box, .gr-panel, .gradio-container .form {
    background: #d8d8d8 !important;
    border: 3px solid #111 !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px #111 !important;
}
/* ── Inputs & Buttons ── */
.gr-textbox, .gr-dropdown, .gr-slider, input, textarea, .gr-dropdown .wrap-inner { 
    border-radius: 0 !important; 
    border: 2px solid #111 !important;
    background: #eee !important;
    color: #111 !important;
    font-weight: bold;
    font-size: 1.1rem !important;
    padding: 8px 12px !important;
}
.gr-button {
    border-radius: 0 !important;
    border: 3px solid #111 !important;
    font-weight: 900 !important;
    text-transform: lowercase;
    transition: all 0.1s ease !important;
    box-shadow: 4px 4px 0px #111 !important;
    color: #111 !important;
    background: transparent !important;
}
.gr-button-primary {
    background: #111 !important;
    color: #d8d8d8 !important;
}
.gr-button:hover, .gr-button-primary:hover {
    box-shadow: 2px 2px 0px #111 !important;
    transform: translate(2px, 2px);
    opacity: 1 !important;
}
/* ── Output tabs ── */
.gr-tabs > div {
    border: 3px solid #111 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: inset 0px 0px 0px #111 !important;
}
.gr-tab-item { 
    border-radius: 0 !important; 
    border: 2px solid transparent !important;
    font-weight: 800 !important;
    text-transform: lowercase;
    color: #555 !important;
}
.gr-tab-item.selected {
    background: #111 !important;
    color: #d8d8d8 !important;
    border: 2px solid #111 !important;
}
/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 0;
    font-size: 0.85rem;
    font-weight: 900;
    background: #111;
    color: #d8d8d8;
    text-transform: lowercase;
    border: 2px solid #111;
}
.gr-examples { margin-top: 12px; }
"""

HEADER_HTML = f"""
<div class="header-box">
    <h1>Multi-Subject Notes</h1>
    <p>Organize, ingest, and query your knowledge base silently isolated per subject</p>
    <br>
    <span class="status-badge">LLM: {_llm_status()}</span>
</div>
"""

with gr.Blocks(title="Multi-Subject Notes") as demo:
    gr.HTML(HEADER_HTML)
    
    # State mapping
    initial_choices = get_subject_choices()
    default_choice = initial_choices[0] if initial_choices else None

    with gr.Tabs() as main_tabs:
        
        # ── Tab 1: Query Subject ──────────────────────────────────────────────
        with gr.Tab("Ask Your Notes", id="query_tab"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes=["card"]):
                    query_subject_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=default_choice,
                        label="Select Subject to Query",
                        interactive=True
                    )
                    
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. Summarize the main points of my notes.",
                        lines=3,
                        elem_id="question_input",
                    )
                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Top-K retrieval candidates",
                            )
                        with gr.Row():
                            category_dropdown = gr.Dropdown(
                                choices=["All", "uploaded", "web", "concept", "architecture"],
                                value="All",
                                label="Filter by category",
                            )
                            reranker_check = gr.Checkbox(
                                value=True,
                                label="Enable cross-encoder reranking",
                            )
                    submit_btn = gr.Button("Ask", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Answer"):
                            answer_output = gr.Markdown(
                                value="*Your answer will appear here after submitting a question.*"
                            )
                        with gr.Tab("Sources"):
                            sources_output = gr.Markdown(
                                value="*Retrieved sources will appear here.*"
                            )
                        with gr.Tab("Eval Metrics"):
                            metrics_output = gr.Markdown(
                                value="*Faithfulness and relevancy scores will appear here.*"
                            )
                        with gr.Tab("Context (Debug)"):
                            context_output = gr.Markdown(
                                value="*Raw context chunks from Endee will appear here.*"
                            )

        # ── Tab 2: Manage Subjects ─────────────────────────────────────────────
        with gr.Tab("Manage Subjects", id="manage_tab"):
            with gr.Row():
                # Column 1: Create New Subject
                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("### Create New Subject")
                    new_subject_input = gr.Textbox(label="Subject Name", placeholder="e.g. Quantum Physics")
                    create_subject_btn = gr.Button("Create Subject")
                    create_subject_msg = gr.Markdown("")
                    
                # Column 2: Ingest into Subject
                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("### Add Content to Subject")
                    ingest_subject_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=default_choice,
                        label="Select Target Subject",
                        interactive=True
                    )
                    
                    upload_files = gr.File(
                        label="Upload text documents (.txt)",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".csv", ".json"]
                    )
                    
                    upload_links = gr.Textbox(
                        label="Or paste URLs (one per line)",
                        lines=3,
                        placeholder="https://example.com/article"
                    )
                    
                    reset_check = gr.Checkbox(
                        label="Clear existing data in this subject before adding?",
                        value=False
                    )
                    
                    ingest_btn = gr.Button("Process & Add Content", variant="primary")
                    ingest_msg = gr.Markdown("")
    
    # ── Event Wiring ────────────────────────────────────────────────────────────
    
    # When tab changes or manually triggered, refresh dropdowns
    main_tabs.select(
        fn=refresh_dropdowns, 
        inputs=None, 
        outputs=[query_subject_dropdown, ingest_subject_dropdown]
    )

    create_subject_btn.click(
        fn=create_subject,
        inputs=[new_subject_input],
        outputs=[create_subject_msg, ingest_subject_dropdown]
    ).then(
        fn=refresh_dropdowns, 
        inputs=None, 
        outputs=[query_subject_dropdown, ingest_subject_dropdown]
    )
    
    ingest_btn.click(
        fn=lambda: "Processing and adding content... Please wait.",
        outputs=[ingest_msg]
    ).then(
        fn=ingest_subject_docs,
        inputs=[ingest_subject_dropdown, upload_files, upload_links, reset_check],
        outputs=[ingest_msg]
    )

    def _show_loading():
        return (
            "Working on your answer... Please wait.",
            "Searching knowledge base...",
            "Computing metrics...",
            "Retrieving context...",
        )

    submit_btn.click(
        fn=_show_loading,
        outputs=[answer_output, sources_output, metrics_output, context_output]
    ).then(
        fn=query_rag,
        inputs=[query_subject_dropdown, question_input, top_k_slider, category_dropdown, reranker_check],
        outputs=[answer_output, sources_output, metrics_output, context_output],
        show_progress="full",
    )
    
    question_input.submit(
        fn=_show_loading,
        outputs=[answer_output, sources_output, metrics_output, context_output]
    ).then(
        fn=query_rag,
        inputs=[query_subject_dropdown, question_input, top_k_slider, category_dropdown, reranker_check],
        outputs=[answer_output, sources_output, metrics_output, context_output],
        show_progress="full",
    )

if __name__ == "__main__":
    logger.info(f"Starting Multi-Subject App on {APP_HOST}:{APP_PORT}")
    demo.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        show_error=True,
        share=False,
        theme=gr.themes.Monochrome(
            font=[gr.themes.GoogleFont("Helvetica Neue"), "Arial", "sans-serif"],
            primary_hue="zinc",
            secondary_hue="slate",
            neutral_hue="stone"
        ),
        css=CUSTOM_CSS,
    )
