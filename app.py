import logging
import os
import sys
import PyPDF2

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

# Global RAG chain instance (lazy-loaded on first query)
_rag_chain: RAGChain | None = None

def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        logger.info("Initialising RAG chain...")
        _rag_chain = RAGChain(use_reranker=True)
    return _rag_chain

# Helper: Format sources and metrics

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
    faith_pct = int(faithfulness * 100)
    rel_pct = int(relevancy * 100)
    return (
        f"| Metric | Score | Visual |\n"
        f"|---|---|---|\n"
        f"| **Faithfulness** | `{faithfulness:.2f}` | {'█' * (faith_pct // 10)}{'░' * (10 - faith_pct // 10)} {faith_pct}% |\n"
        f"| **Answer Relevancy** | `{relevancy:.2f}` | {'█' * (rel_pct // 10)}{'░' * (10 - rel_pct // 10)} {rel_pct}% |\n"
        f"| **Latency** | `{latency:.2f}s` | — |\n"
    )

def _llm_status() -> str:
    provider = LLM_PROVIDER.lower()
    if provider == "openai":
        if OPENAI_API_KEY:
            return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
        return "Warning: OpenAI — no API key set"
    elif provider == "google":
        if GOOGLE_API_KEY:
            return f"Google Gemini ({os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash')})"
        return "Warning: Google Gemini — no API key set"
    elif provider == "ollama":
        return f"Ollama ({OLLAMA_MODEL})"
    return f"Unknown: {LLM_PROVIDER}"

# Core Logic

def get_subject_choices():
    return list(load_subjects().keys())

def create_subject(name):
    if not name or not name.strip():
        return "Warning: Please enter a subject name.", gr.update()
    
    name = name.strip()
    create_or_get_subject(name)
    choices = get_subject_choices()
    
    return f"Success: Subject **'{name}'** created successfully!", gr.update(choices=choices, value=name)

def ingest_subject_docs(subject, files, links_text, reset):
    if not subject:
        return "Warning: Please select a subject first."
        
    index_name = get_subject_index_name(subject)
    if not index_name:
        return f"Error: Internal error: Index for subject '{subject}' not found."
        
    docs = []
    if files:
        for f in files:
            try:
                if f.name.lower().endswith('.pdf'):
                    with open(f.name, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        content = ""
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                        docs.append({"title": os.path.basename(f.name), "content": content})
                else:
                    with open(f.name, "r", encoding="utf-8") as file:
                        content = file.read()
                        docs.append({"title": os.path.basename(f.name), "content": content})
            except Exception as e:
                logger.error(f"Error reading file {f.name}: {e}")
                
    links = [l.strip() for l in (links_text or "").split("\n") if l.strip()]
    
    if not docs and not links and not reset:
        return "Warning: No documents, links, or reset requested."
        
    try:
        count = ingest_documents(index_name=index_name, docs=docs, links=links, reset=reset)
        return f"Success: Ingestion complete for **'{subject}'**! **{count}** vectors indexed."
    except Exception as e:
        logger.exception("Ingest failed")
        return f"Error: Ingestion failed: {str(e)}"

def query_rag(subject, question, top_k, category_filter, use_reranker):
    if not subject:
        return "Warning: Please select a subject first.", "", "", ""
        
    if not question or not question.strip():
        return "Warning: Please enter a question.", "", "", ""

    index_name = get_subject_index_name(subject)
    if not index_name:
         return f"Error: Internal error: Index for subject '{subject}' not found.", "", "", ""

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

        answer_md = response.answer
        sources_md = _format_sources(response.sources)
        metrics_md = _format_metrics(response.faithfulness, response.answer_relevancy, response.latency_seconds)
        context_md = (
            f"```\n{response.context[:3000]}\n```"
            if response.context
            else "*No context retrieved.*"
        )

        return answer_md, sources_md, metrics_md, context_md

    except Exception as e:
        logger.exception("Error in RAG query")
        return (
            f"**Error:** {str(e)}\n\nCheck that Endee is running.",
            "",
            "",
            "",
        )

def refresh_dropdowns():
    choices = get_subject_choices()
    val = choices[0] if choices else None
    return gr.update(choices=choices, value=val), gr.update(choices=choices, value=val)

# UI Construction

CUSTOM_CSS = """
/* ── CSS Variables ── */

/* ── CSS Variables ── */
:root {
    --bg-primary: #F0EDE5; /* Requested off-white */
    --accent: #004643;     /* Requested dark teal */
    --bg-paper: #ffffff;
    --text-primary: #004643;
    --text-secondary: #004643;
    
    /* Neobrutalism Specs */
    --border-width: 3px;
    --radius: 4px;
    --shadow-offset: 5px;
    --shadow-brutal: var(--shadow-offset) var(--shadow-offset) 0px #004643;
    --shadow-hover: 2px 2px 0px #004643;
    
    --transition: all 0.1s ease-in-out;
}

/* ── Global Resets ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Hide footer ── */
footer, .gradio-container footer { display: none !important; }

/* ── Header ── */
.app-header {
    background: var(--bg-primary);
    border: var(--border-width) solid var(--accent);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-brutal);
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent) !important;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: -0.5px;
}
.app-header p {
    color: var(--accent) !important;
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 0;
    font-size: 0.85rem;
    font-weight: 800;
    background: #fdf577; /* Bright yellow for contrast */
    color: var(--accent);
    border: 2px solid var(--accent);
    margin-top: 16px;
    box-shadow: 2px 2px 0px var(--accent);
    text-transform: uppercase;
}

/* ── Tabs ── */
.gr-tab-item {
    border-radius: var(--radius) var(--radius) 0 0 !important;
    border: var(--border-width) solid var(--accent) !important;
    border-bottom: none !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
    background: var(--bg-primary) !important;
    padding: 12px 24px !important;
    transition: var(--transition) !important;
    text-transform: uppercase;
    margin-right: -3px !important; /* Overlap borders */
}
.gr-tab-item:hover {
    background: #e4e0d5 !important;
}
.gr-tab-item.selected {
    background: var(--bg-paper) !important;
    box-shadow: 0 -4px 0px var(--accent) inset !important;
}

/* ── Cards / Content Areas ── */
.panel-card, .gr-form, .gr-box {
    background: var(--bg-paper) !important;
    border: var(--border-width) solid var(--accent) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    box-shadow: var(--shadow-brutal) !important;
}

/* ── Inputs ── */
.gr-textbox textarea, .gr-textbox input,
input[type="text"], textarea {
    background: var(--bg-paper) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 0 !important;
    color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 14px !important;
    box-shadow: 3px 3px 0px var(--accent) !important;
    transition: var(--transition) !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus,
input[type="text"]:focus, textarea:focus {
    box-shadow: 1px 1px 0px var(--accent) !important;
    transform: translate(2px, 2px) !important;
    outline: none !important;
}

/* ── Dropdowns ── */
.gr-dropdown, .gr-dropdown .wrap-inner {
    background: var(--bg-paper) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 0 !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: 3px 3px 0px var(--accent) !important;
}

/* ── Buttons ── */
.gr-button {
    border-radius: 0 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    text-transform: uppercase;
    padding: 12px 24px !important;
    border: var(--border-width) solid var(--accent) !important;
    background: var(--bg-paper) !important;
    color: var(--accent) !important;
    box-shadow: var(--shadow-brutal) !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
}
.gr-button:hover {
    background: #e4e0d5 !important;
}
.gr-button:active {
    box-shadow: var(--shadow-hover) !important;
    transform: translate(3px, 3px) !important;
}
button.primary, .gr-button-primary, #ask_btn, #create_subj_btn, #ingest_btn {
    background: #fdf577 !important;
    color: var(--accent) !important;
}
button.primary:hover, .gr-button-primary:hover, #ask_btn:hover, #create_subj_btn:hover, #ingest_btn:hover {
    background: #fce83a !important; /* Slightly darker yellow */
}

/* ── Form labels ── */
label, .gr-box label, .gr-textbox label, .gr-dropdown label {
    color: var(--accent) !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em !important;
}

/* ── Markdown output ── */
.prose, .markdown-text, .gr-markdown {
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
}
.prose h1, .prose h2, .prose h3 {
    color: var(--accent) !important;
    font-weight: 900 !important;
    text-transform: uppercase;
}
.prose code, .gr-markdown code {
    background: #fdf577 !important;
    color: var(--accent) !important;
    padding: 2px 6px !important;
    border: 2px solid var(--accent) !important;
    font-weight: 700 !important;
}
.prose pre, .gr-markdown pre {
    background: var(--bg-paper) !important;
    border: var(--border-width) solid var(--accent) !important;
    border-radius: 0 !important;
    padding: 16px !important;
    box-shadow: 4px 4px 0px var(--accent) !important;
}

/* ── Accordion ── */
.gr-accordion {
    border: var(--border-width) solid var(--accent) !important;
    border-radius: 0 !important;
    background: var(--bg-paper) !important;
    box-shadow: 4px 4px 0px var(--accent) !important;
}
.gr-accordion .label-wrap {
    color: var(--accent) !important;
    font-weight: 800 !important;
    text-transform: uppercase;
}

/* ── File Upload ── */
.gr-file, .gr-file .wrap {
    background: var(--bg-primary) !important;
    border: var(--border-width) dashed var(--accent) !important;
    border-radius: 0 !important;
}

/* ── Examples ── */
.gr-examples .gr-sample-textbox {
    background: var(--bg-paper) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 0 !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: 2px 2px 0px var(--accent) !important;
}

/* ── Section title ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 900;
    color: var(--accent);
    margin-bottom: 16px;
    text-transform: uppercase;
}

/* ── Tables ── */
.prose table, .gr-markdown table {
    border-collapse: collapse !important;
    border: var(--border-width) solid var(--accent) !important;
}
.prose th, .gr-markdown th {
    background: #fdf577 !important;
    color: var(--accent) !important;
    font-weight: 800 !important;
    padding: 12px !important;
    border: 2px solid var(--accent) !important;
    text-transform: uppercase;
}
.prose td, .gr-markdown td {
    padding: 12px !important;
    border: 2px solid var(--accent) !important;
    font-weight: 600 !important;
}
"""

HEADER_HTML = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800;900&display=swap" rel="stylesheet">
<div class="app-header">
    <h1>Multi-Subject Notes</h1>
    <p>Organize, ingest, and query your knowledge base — each subject is isolated with its own vector index.</p>
    <br>
    <div class="status-chip">
        {_llm_status()}
    </div>
</div>
"""

EXAMPLE_QUESTIONS = [
    "Summarize the main points of my notes.",
    "What are the key concepts covered?",
    "Explain the most important topics in detail.",
    "What connections exist between different ideas?",
]

with gr.Blocks(title="Multi-Subject Notes") as demo:
    gr.HTML(HEADER_HTML)
    
    # State mapping
    initial_choices = get_subject_choices()
    default_choice = initial_choices[0] if initial_choices else None

    with gr.Tabs() as main_tabs:
        
        # ── Tab 1: Ask Your Notes ─────────────────────────────────────────
        with gr.Tab("Ask Your Notes", id="query_tab"):
            with gr.Row(equal_height=True):
                # Left panel: Input
                with gr.Column(scale=2, elem_classes=["panel-card"]):
                    gr.HTML('<div class="section-title">Subject</div>')
                    query_subject_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=default_choice,
                        label="Subject",
                        interactive=True,
                    )
                    
                    gr.HTML('<div class="section-title" style="margin-top:20px;">Your Question</div>')
                    question_input = gr.Textbox(
                        lines=4,
                        placeholder="Ask anything about your notes...",
                        label="",
                        elem_id="question_input",
                    )

                    gr.HTML('<div style="margin-top:16px; margin-bottom:4px; font-size:0.8rem; color:var(--text-secondary); font-weight:600; text-transform:uppercase;">Try an example</div>')
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=question_input,
                        label="",
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
                    submit_btn = gr.Button("Ask", variant="primary", size="lg", elem_id="ask_btn")

                # Right panel: Results
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Answer"):
                            answer_output = gr.Markdown(
                                value="Your answer will appear here after submitting a question."
                            )
                        with gr.Tab("Sources"):
                            sources_output = gr.Markdown(
                                value="Retrieved sources will appear here."
                            )
                        with gr.Tab("Metrics"):
                            metrics_output = gr.Markdown(
                                value="Faithfulness and relevancy scores will appear here."
                            )
                        with gr.Tab("Context"):
                            context_output = gr.Markdown(
                                value="Raw context chunks from Endee will appear here."
                            )

        # ── Tab 2: Manage Subjects ────────────────────────────────────────
        with gr.Tab("Manage Subjects", id="manage_tab"):
            with gr.Row(equal_height=True):
                # Column 1: Create New Subject
                with gr.Column(elem_classes=["panel-card"]):
                    gr.HTML('<div class="section-title">Create New Subject</div>')
                    new_subject_input = gr.Textbox(
                        label="Subject Name",
                        placeholder="e.g. Quantum Physics, History, Biology",
                    )
                    create_subject_btn = gr.Button("Create Subject", variant="primary", elem_id="create_subj_btn")
                    create_subject_msg = gr.Markdown("")
                    
                # Column 2: Ingest into Subject
                with gr.Column(elem_classes=["panel-card"]):
                    gr.HTML('<div class="section-title">Add Content to Subject</div>')
                    ingest_subject_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        value=default_choice,
                        label="Target Subject",
                        interactive=True,
                    )
                    
                    upload_files = gr.File(
                        label="Upload documents (.txt, .pdf, .md, .csv, .json)",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".csv", ".json", ".pdf"],
                    )
                    
                    upload_links = gr.Textbox(
                        label="Or paste URLs (one per line)",
                        lines=3,
                        placeholder="https://example.com/article",
                    )
                    
                    reset_check = gr.Checkbox(
                        label="Clear existing data before adding?",
                        value=False,
                    )
                    
                    ingest_btn = gr.Button("Process & Add Content", variant="primary", elem_id="ingest_btn")
                    ingest_msg = gr.Markdown("")
    
    # ── Event Wiring ────────────────────────────────────────────────────────
    
    main_tabs.select(
        fn=refresh_dropdowns, 
        inputs=None, 
        outputs=[query_subject_dropdown, ingest_subject_dropdown],
    )

    create_subject_btn.click(
        fn=create_subject,
        inputs=[new_subject_input],
        outputs=[create_subject_msg, ingest_subject_dropdown],
    ).then(
        fn=refresh_dropdowns, 
        inputs=None, 
        outputs=[query_subject_dropdown, ingest_subject_dropdown],
    )
    
    ingest_btn.click(
        fn=lambda: "Processing and adding content... Please wait.",
        outputs=[ingest_msg],
    ).then(
        fn=ingest_subject_docs,
        inputs=[ingest_subject_dropdown, upload_files, upload_links, reset_check],
        outputs=[ingest_msg],
    )

    def _show_loading():
        return (
            "⏳ Generating your answer...",
            "⏳ Searching knowledge base...",
            "⏳ Computing metrics...",
            "⏳ Retrieving context...",
        )

    submit_btn.click(
        fn=_show_loading,
        outputs=[answer_output, sources_output, metrics_output, context_output],
    ).then(
        fn=query_rag,
        inputs=[query_subject_dropdown, question_input, top_k_slider, category_dropdown, reranker_check],
        outputs=[answer_output, sources_output, metrics_output, context_output],
        show_progress="full",
    )
    
    question_input.submit(
        fn=_show_loading,
        outputs=[answer_output, sources_output, metrics_output, context_output],
    ).then(
        fn=query_rag,
        inputs=[query_subject_dropdown, question_input, top_k_slider, category_dropdown, reranker_check],
        outputs=[answer_output, sources_output, metrics_output, context_output],
        show_progress="full",
    )

# JavaScript to forcefully disable dark mode and force light mode
FORCE_LIGHT_MODE_JS = """
function() {
    document.querySelector('body').classList.remove('dark');
    document.querySelector('body').classList.add('light');
    
    // Add logic to continually observe and remove the dark class if re-added
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === "class") {
                if (document.body.classList.contains('dark')) {
                    document.body.classList.remove('dark');
                    document.body.classList.add('light');
                }
            }
        });
    });
    observer.observe(document.body, { attributes: true });
}
"""

if __name__ == "__main__":
    logger.info(f"Starting Multi-Subject Notes on {APP_HOST}:{APP_PORT}")
    demo.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        show_error=True,
        share=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        js=FORCE_LIGHT_MODE_JS,
    )
