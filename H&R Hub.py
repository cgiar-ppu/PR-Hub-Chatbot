import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from openai import OpenAI

# --- CGIAR Theme --------------------------------------------------------------
import html  # to escape text in chips

import pandas as pd
import pickle
import hashlib
from dataclasses import asdict

CGIAR_COLORS = {
    "green_primary": "#427730",      # Corporate Green
    "green_leaf": "#7AB800",         # Leaf green
    "green_leaf_dark": "#739600",    # Darker leaf green
    "blue_bright": "#0065BD",        # Bright Blue
    "blue_medium": "#0039A6",        # Medium Blue
    "yellow": "#FDC82F",             # Yellow
    "orange": "#E37222",             # Orange (amber-ish)
    "bg": "#F7FAF8",                 # Light soft background
    "panel": "#FFFFFF",              # Cards
    "text": "#1A202C",               # Main text
    "muted": "#4A5568",              # Secondary text
    "border": "#E2E8F0",             # Subtle borders
}

def apply_cgiar_theme():
    st.markdown(f"""
    <style>
        /* Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {{
            --brand-primary: {CGIAR_COLORS["green_primary"]};
            --brand-primary-strong: {CGIAR_COLORS["green_leaf_dark"]};
            --brand-accent: {CGIAR_COLORS["green_leaf"]};
            --brand-blue: {CGIAR_COLORS["blue_bright"]};
            --brand-blue-strong: {CGIAR_COLORS["blue_medium"]};
            --brand-yellow: {CGIAR_COLORS["yellow"]};
            --brand-orange: {CGIAR_COLORS["orange"]};

            --bg: {CGIAR_COLORS["bg"]};
            --panel: {CGIAR_COLORS["panel"]};
            --text: {CGIAR_COLORS["text"]};
            --muted: {CGIAR_COLORS["muted"]};
            --border: {CGIAR_COLORS["border"]};
        }}

        .stApp {{
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
        }}
        #MainMenu {{ display: none; }}
        footer {{ visibility: hidden; }}

        /* Main container */
        .main .block-container {{
            max-width: 1100px;
            padding-top: 1.25rem;
        }}

        /* Hero — solid amber */
        .brand-hero {{
            background: var(--brand-orange);
            color: white;
            border-radius: 12px;
            padding: 1.25rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }}
        .brand-hero h1 {{
            margin: 0 0 .25rem 0;
            font-weight: 700;
            letter-spacing: .2px;
        }}
        .brand-hero p {{
            margin: 0;
            opacity: .95;
        }}

        /* Cards */
        .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            margin: .75rem 0;
        }}
        .answer-card {{
            border-left: 4px solid var(--brand-primary);
        }}

        /* Source chips */
        .sources-wrap {{
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: .5rem;
        }}
        .source-chip {{
            background: #f2f7f3;
            border: 1px solid #e1efe4;
            color: #0f3b1f;
            border-radius: 999px;
            padding: .35rem .75rem;
            font-size: .85rem;
            line-height: 1;
            white-space: nowrap;
        }}

        /* Buttons */
        .stButton > button {{
            width: 100%;
            border: 0;
            border-radius: 10px;
            font-weight: 600;
            padding: .65rem 1rem;
            transition: transform .08s ease, opacity .15s ease, box-shadow .2s ease;
            background: var(--brand-primary);
            color: #fff;
            box-shadow: 0 2px 6px rgba(66,119,48,.20);
        }}
        .stButton > button:hover {{
            opacity: .95;
            box-shadow: 0 4px 10px rgba(66,119,48,.25);
            transform: translateY(-1px);
        }}
        .stButton > button:active {{
            transform: translateY(1px);
            box-shadow: inset 0 2px 4px rgba(0,0,0,.08);
        }}

        /* Inputs */
        .stTextInput > div > div > input {{
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
        }}
        .stTextInput > div > div > input:focus {{
            border-color: var(--brand-primary) !important;
            outline: 3px solid rgba(66,119,48,.15) !important;
        }}

        /* Slider */
        .stSlider [data-baseweb="slider"] > div:first-child {{
            color: var(--brand-primary);
        }}

        /* Subtle alerts */
        .stAlert {{
            border-left: 4px solid var(--brand-accent);
        }}

        /* Compact metrics */
        .metric-row > div > div {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: .75rem;
        }}

        /* Footer */
        .app-footer {{
            text-align: center;
            color: var(--muted);
            font-size: .9rem;
            margin: 1rem 0 2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

@dataclass
class Chunk:
    text: str
    source_path: str
    source_name: str
    kind: str  # pdf | docx | pptx
    location: str  # e.g., "page 3", "slide 2", "section X / paragraph 12"
    id: str = ""


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if len(p) > 0:
            sentences.append(p)
    if len(sentences) <= 1:
        lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
        if len(lines) > len(sentences):
            sentences = lines
    return sentences


def read_pdf_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        reader = PdfReader(path)
        for index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            cleaned = normalize_whitespace(page_text)
            if cleaned:
                chunks.append(
                    Chunk(
                        text=cleaned,
                        source_path=path,
                        source_name=os.path.basename(path),
                        kind="pdf",
                        location=f"page {index}",
                        id=f"pdf:{os.path.basename(path)}:p{index}",
                    )
                )
    except Exception:
        pass
    return chunks


def read_docx_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        doc = DocxDocument(path)
        current_section: Optional[str] = None
        for paragraph_index, paragraph in enumerate(doc.paragraphs, start=1):
            text = normalize_whitespace(paragraph.text)
            if not text:
                continue
            style_name = getattr(paragraph.style, "name", "") or ""
            if style_name.lower().startswith("heading") or style_name.lower().startswith("título"):
                current_section = text
            location = (
                f"section '{current_section}'" if current_section else f"paragraph {paragraph_index}"
            )
            chunks.append(
                Chunk(
                    text=text,
                    source_path=path,
                    source_name=os.path.basename(path),
                    kind="docx",
                    location=location,
                    id=f"docx:{os.path.basename(path)}:p{paragraph_index}",
                )
            )
    except Exception:
        pass
    return chunks


def read_pptx_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        prs = Presentation(path)
        for slide_index, slide in enumerate(prs.slides, start=1):
            texts: List[str] = []
            for shape in slide.shapes:
                try:
                    if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                        txt = "\n".join(p.text for p in shape.text_frame.paragraphs)
                        txt = normalize_whitespace(txt)
                        if txt:
                            texts.append(txt)
                except Exception:
                    continue
            slide_text = normalize_whitespace("\n".join(texts))
            if slide_text:
                chunks.append(
                    Chunk(
                        text=slide_text,
                        source_path=path,
                        source_name=os.path.basename(path),
                        kind="pptx",
                        location=f"slide {slide_index}",
                        id=f"pptx:{os.path.basename(path)}:s{slide_index}",
                    )
                )
    except Exception:
        pass
    return chunks


def load_corpus(root_dir: str,
                hide_failures: bool = True,
                exclude_patterns: Optional[List[str]] = None) -> List[Chunk]:
    """
    Recorre root_dir y subcarpetas, carga solo .pdf/.docx/.pptx.
    Muestra únicamente los archivos cargados con éxito (por defecto).
    Puedes excluir archivos por patrón (p.ej., ['^default\\.', '^\\._', '^~\\$']).
    """
    supported_ext = {".pdf", ".docx", ".pptx"}
    exclude_patterns = exclude_patterns or []
    compiled_excludes = [re.compile(pat, re.IGNORECASE) for pat in exclude_patterns]

    chunks: List[Chunk] = []
    ok_files: List[Tuple[str, int]] = []   # (nombre, n_chunks)
    bad_files: List[str] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            # Excluir por patrón (ej.: default.*, archivos ocultos de macOS, backups temporales)
            if any(p.match(fname) for p in compiled_excludes):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_ext:
                continue

            abspath = os.path.join(dirpath, fname)
            # Excluir archivos vacíos
            try:
                if os.path.getsize(abspath) == 0:
                    bad_files.append(fname)
                    continue
            except Exception:
                bad_files.append(fname)
                continue

            new_chunks: List[Chunk] = []
            try:
                if ext == ".pdf":
                    new_chunks = read_pdf_chunks(abspath)
                elif ext == ".docx":
                    new_chunks = read_docx_chunks(abspath)
                elif ext == ".pptx":
                    new_chunks = read_pptx_chunks(abspath)
            except Exception:
                new_chunks = []

            if new_chunks:
                chunks.extend(new_chunks)
                ok_files.append((fname, len(new_chunks)))
            else:
                bad_files.append(fname)

    # Panel de “detalles de procesamiento”
    with st.expander("🔍 View document processing details", expanded=False):
        st.write("### ✅ Loaded files:")
        for fname, n in ok_files:
            st.write(f"✅ {fname} ({n} chunks)")
        st.write(f"\n**Summary:** {len(ok_files)} loaded")
        if not hide_failures and bad_files:
            st.write("### ❌ Skipped/failed files:")
            for fname in bad_files:
                st.write(f"❌ {fname}")

    return chunks


def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, any]:
    texts = [c.text for c in chunks]
    if not texts:
        vectorizer = TfidfVectorizer(stop_words=None)
        vectorizer.fit(["dummy"])
        matrix = vectorizer.transform(["dummy"])
        return vectorizer, matrix
    vectorizer = TfidfVectorizer(stop_words=None, max_df=0.9)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def get_corpus_hash(root_dir: str) -> str:
    hash_str = ''
    supported_ext = {".pdf", ".docx", ".pptx"}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() not in supported_ext:
                continue
            abspath = os.path.join(dirpath, fname)
            mtime = os.path.getmtime(abspath)
            size = os.path.getsize(abspath)
            hash_str += f"{abspath}:{mtime}:{size}\n"
    return hashlib.sha256(hash_str.encode()).hexdigest()

def rank_chunks(query: str, vectorizer: TfidfVectorizer, matrix, chunks: List[Chunk], top_k: int = 25) -> List[Tuple[Chunk, float]]:
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idx_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    results: List[Tuple[Chunk, float]] = []
    for idx, score in idx_scores[: max(top_k * 2, top_k)]:
        if score <= 0.02:
            continue
        results.append((chunks[idx], float(score)))
        if len(results) >= top_k:
            break
    return results


def extract_relevant_sentences(query: str, texts: List[str], max_sentences: int = 6) -> List[str]:
    query_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    candidates: List[Tuple[str, float]] = []
    for block in texts:
        for sent in split_sentences(block):
            low = sent.lower()
            if not low:
                continue
            tf = sum(low.count(t) for t in query_terms)
            if tf == 0:
                continue
            length_penalty = 1.0 + max(0, (len(sent) - 300) / 300.0)
            score = tf / length_penalty
            candidates.append((sent.strip(), score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    unique: List[str] = []
    seen = set()
    for sent, _ in candidates:
        key = sent[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(sent)
        if len(unique) >= max_sentences:
            break
    return unique


def compose_answer(query: str, ranked: List[Tuple[Chunk, float]]) -> Tuple[str, List[str]]:
    if not ranked:
        msg = (
            "Not found in the available information. "
            "A specific reference (document/page or section) would be needed. "
            "Verify the document name or try other keywords."
        )
        return msg, []
    texts = [c.text for c, _ in ranked]
    sentences = extract_relevant_sentences(query, texts, max_sentences=6)
    if len(sentences) == 0:
        msg = (
            "Not found in the available information. "
            "There are no relevant snippets for the current query."
        )
        return msg, []
    if len(sentences) < 3 and len(ranked) >= 2:
        extra_sentences = []
        for c, _ in ranked:
            sents = split_sentences(c.text)
            for s in sents:
                if len(s.strip()) > 40 and s.strip() not in sentences:
                    extra_sentences.append(s.strip())
                if len(sentences) + len(extra_sentences) >= 3:
                    break
            if len(sentences) + len(extra_sentences) >= 3:
                break
        sentences.extend(extra_sentences[: max(0, 3 - len(sentences))])
    sentences = sentences[:6]
    answer = " ".join(sentences).strip()
    return answer, [f"{c.source_name} — {c.location} — {c.id}" for c, _ in ranked]


def format_sources_lines(ranked: List[Tuple[Chunk, float]] , max_items: int = 10) -> List[str]:
    lines: List[str] = []
    seen = set()
    for c, _ in ranked:
        entry = f"{c.source_name} — {c.location} — {c.id}"
        if entry in seen:
            continue
        seen.add(entry)
        lines.append(entry)
        if len(lines) >= max_items:
            break
    return lines


def call_openai_generate(query: str, ranked: List[Tuple[Chunk, float]], max_sentences: int = 5) -> Optional[str]:
    max_ctx = 12
    selected = ranked[:max_ctx]
    context_blocks: List[str] = []
    for c, _ in selected:
        context_blocks.append(
            f"[ID: {c.id}]\nFile: {c.source_name}\nLocation: {c.location}\nContent: {c.text}"
        )
    context = "\n\n---\n\n" + "\n\n---\n\n".join(context_blocks) if context_blocks else ""

    system_msg = (
        "You are a RAG assistant. Use ONLY the provided context as your source, "
        "without copying full sentences verbatim from chunks (max 10 consecutive words). "
        "Write clearly and cohesively, interpreting the usage context to adapt the response. "
        "Do not invent or extrapolate beyond the context and preserve acronyms EXACTLY as written. "
        "ALWAYS answer in English, regardless of the user's language. "
        "If there is insufficient evidence, return EXACTLY: 'I cannot find information in the provided chunks to answer this.'"
    )

    user_msg = (
        "Output instructions:\n"
        "- 3 to 5 sentences, neutral and direct style, no lists.\n"
        "- ALWAYS answer in English and adapt wording to the question's context.\n"
        "- End with the literal 'Sources:' and then, as a list, each line as 'File — Location — Cited IDs'.\n"
        "- If insufficient evidence, return EXACTLY: 'I cannot find information in the provided chunks to answer this.'\n\n"
        f"Question: {query}\n\nContext:{context}"
    )

    try:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            try:
                resp2 = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_output_tokens=500,
                )
                if hasattr(resp2, "output") and resp2.output and hasattr(resp2.output[0], "content"):
                    parts = resp2.output[0].content
                    if parts and hasattr(parts[0], "text"):
                        return (parts[0].text or "").strip()
            except Exception:
                return None
    except Exception:
        return None

    return None


def dedupe_preserve_order(items: List[str], limit: int = 5) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
        if len(out) >= limit:
            break
    return out

def render_sources_pills(lines):
    if not lines:
        st.markdown("<div class='sources-wrap'><span class='source-chip'>not specified</span></div>", unsafe_allow_html=True)
        return
    pills = "".join(f"<span class='source-chip'>{html.escape(line)}</span>" for line in lines)
    st.markdown(f"<div class='sources-wrap'>{pills}</div>", unsafe_allow_html=True)

def render_app() -> None:
    st.set_page_config(page_title="P&R Hub — RAG (CGIAR)", page_icon="🌿", layout="centered")
    apply_cgiar_theme()

    # Hero header (solid amber + new title)
    st.markdown("""
        <div class="brand-hero">
            <h1>P&R Hub — Document-grounded RAG assistant</h1>
            <p>Answers strictly based on the documents in this project. Always include the <strong>Sources</strong> section.</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar: quick help & example (English, with help icon, and your example)
    with st.sidebar:
        st.markdown("### ❓ Quick help")
        st.write("- Place your **PDF/DOCX/PPTX** in the project folder.")
        st.write("- Ask a specific question.")
        st.write("- Adjust the number of chunks if you need more/less context.")
        st.markdown("---")
        st.markdown("**Examples**")
        st.caption("• What evidence architecture and rigor standards underpin the report’s claims? Summarize the dataset, methods, and how to interpret causality.")
        st.markdown("---")

    project_root = os.path.dirname(os.path.abspath(__file__))
    chunks_file = os.path.join(project_root, 'chunks.xlsx')
    index_file = os.path.join(project_root, 'index.pkl')
    hash_file = os.path.join(project_root, 'corpus_hash.txt')

    current_hash = get_corpus_hash(project_root)

    load_from_cache = False
    if all(os.path.exists(f) for f in [hash_file, chunks_file, index_file]):
        with open(hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            load_from_cache = True

    if load_from_cache:
        with st.spinner("Loading from cache..."):
            df = pd.read_excel(chunks_file)
            chunks = [Chunk(**row) for row in df.to_dict(orient='records')]
            with open(index_file, 'rb') as f:
                data = pickle.load(f)
            vectorizer = data['vectorizer']
            matrix = data['matrix']
    else:
        with st.spinner("Loading documents..."):
            chunks = load_corpus(project_root)
        df = pd.DataFrame([asdict(c) for c in chunks])
        df.to_excel(chunks_file, index=False)
        with st.spinner("Building index..."):
            vectorizer, matrix = build_index(chunks)
        with open(index_file, 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'matrix': matrix}, f)
        with open(hash_file, 'w') as f:
            f.write(current_hash)

    num_docs = len({c.source_path for c in chunks})
    num_chunks = len(chunks)

    # Metric cards
    cols = st.columns(2, gap="small")

    with cols[0]:
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:.65rem">
        <span style="font-size:1.35rem">📄</span>
        <div>
            <div style="font-size:.8rem;color:var(--muted)">Loaded documents</div>
            <div style="font-weight:700">{num_docs}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="card" style="display:flex;align-items:center;gap:.65rem">
        <span style="font-size:1.35rem">🧩</span>
        <div>
            <div style="font-size:.8rem;color:var(--muted)">Indexed chunks</div>
            <div style="font-weight:700">{num_chunks}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    if num_chunks == 0:
        st.warning(
            "No compatible documents (.pdf, .docx, .pptx) were found in the project. "
            "Add files to the existing folders and reload."
        )

    # Search area as a form
    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input("Type your question:", value="", placeholder="e.g., What is the grievance procedure timeline?")
        top_k = st.slider("Number of chunks to consider", min_value=20, max_value=100, value=25, help="Higher values = more recall, slightly slower.")
        submitted = st.form_submit_button("🔎 Search", use_container_width=True)

    if submitted:
        ranked = rank_chunks(query, vectorizer, matrix, chunks, top_k=top_k)

        # Try OpenAI (if API key present) per your rules
        ai_answer = call_openai_generate(query, ranked, max_sentences=5)

        # Answer card
        if ai_answer is None or not ai_answer.strip():
            answer, sources_all = compose_answer(query, ranked)

            # Keep answer compact
            unavailable = answer.startswith("I cannot find information in the provided chunks to answer this.") or \
                answer.startswith("Not found in the available information.")
            if not unavailable:
                sents = split_sentences(answer)
                if len(sents) > 6:
                    answer = " ".join(sents[:6]).strip()

            st.markdown(f"<div class='card answer-card'>{html.escape(answer)}</div>", unsafe_allow_html=True)

            # Source chips (de-duplicated, max 10)
            render_sources_pills(dedupe_preserve_order(sources_all, limit=10))
        else:
            # Model already returns text + "Sources:"; present it in a card
            safe = html.escape(ai_answer).replace("\n", "<br>")
            st.markdown(f"<div class='card answer-card'>{safe}</div>", unsafe_allow_html=True)

            # Reinforce sources section with our TF-IDF ranking
            st.caption("Top sources (from TF-IDF ranking):")
            render_sources_pills(format_sources_lines(ranked, max_items=10))

    # Footer
    st.markdown(
        "<div class='app-footer'>Prototype · CGIAR-inspired UI · © 2025</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    render_app()