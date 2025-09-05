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


@dataclass
class Chunk:
    text: str
    source_path: str
    source_name: str
    kind: str  # pdf | docx | pptx
    location: str  # e.g., "página 3", "diapositiva 2", "sección X / párrafo 12"
    id: str = ""


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Simple sentence splitter suitable for ES text; avoids splitting on abbreviations naively
    # Falls back to newline when punctuation is scarce
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if len(p) > 0:
            sentences.append(p)
    if len(sentences) <= 1:
        # Try line-based splitting if punctuation is rare
        lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
        if len(lines) > len(sentences):
            sentences = lines
    return sentences


def read_pdf_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        reader = PdfReader(path)
        for index, page in enumerate(reader.pages):
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
                        location=f"página {index + 1}",
                        id=f"pdf:{os.path.basename(path)}:p{index + 1}",
                    )
                )
    except Exception:
        # Skip unreadable PDFs silently
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
                f"sección '{current_section}'" if current_section else f"párrafo {paragraph_index}"
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
                        location=f"diapositiva {slide_index}",
                        id=f"pptx:{os.path.basename(path)}:s{slide_index}",
                    )
                )
    except Exception:
        pass
    return chunks


def load_corpus(root_dir: str) -> List[Chunk]:
    supported_ext = {".pdf", ".docx", ".pptx"}
    chunks: List[Chunk] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_ext:
                continue
            abspath = os.path.join(dirpath, fname)
            if ext == ".pdf":
                chunks.extend(read_pdf_chunks(abspath))
            elif ext == ".docx":
                chunks.extend(read_docx_chunks(abspath))
            elif ext == ".pptx":
                chunks.extend(read_pptx_chunks(abspath))
    return chunks


@st.cache_resource(show_spinner=False)
def build_index(chunks: List[Chunk]) -> Tuple[TfidfVectorizer, any]:
    texts = [c.text for c in chunks]
    if not texts:
        # Fit a dummy vectorizer to avoid runtime errors; it will simply match nothing
        vectorizer = TfidfVectorizer(stop_words=None)
        vectorizer.fit(["dummy"])
        matrix = vectorizer.transform(["dummy"])
        return vectorizer, matrix
    vectorizer = TfidfVectorizer(stop_words=None, max_df=0.9)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def rank_chunks(query: str, vectorizer: TfidfVectorizer, matrix, chunks: List[Chunk], top_k: int = 25) -> List[Tuple[Chunk, float]]:
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idx_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    # Filter very low-similarity noise
    results: List[Tuple[Chunk, float]] = []
    for idx, score in idx_scores[: max(top_k * 2, top_k)]:
        if score <= 0.02:
            continue
        results.append((chunks[idx], float(score)))
        if len(results) >= top_k:
            break
    return results


def extract_relevant_sentences(query: str, texts: List[str], max_sentences: int = 6) -> List[str]:
    # Keyword-based scoring to stay extractive and faithful
    query_terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    candidates: List[Tuple[str, float]] = []
    for block in texts:
        for sent in split_sentences(block):
            low = sent.lower()
            if not low:
                continue
            # Score: term frequency + length normalization preference
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
            "No se encuentra en la información disponible. "
            "Faltaría una referencia específica (documento/página o sección). "
            "Verifica el nombre del documento o intenta con otras palabras clave."
        )
        return msg, []
    texts = [c.text for c, _ in ranked]
    sentences = extract_relevant_sentences(query, texts, max_sentences=6)
    # Enforce 3–6 sentences if possible
    if len(sentences) == 0:
        msg = (
            "No se encuentra en la información disponible. "
            "No hay fragmentos relevantes para la consulta actual."
        )
        return msg, []
    if len(sentences) < 3 and len(ranked) >= 2:
        # If too few sentences, append short neutral connectors from top texts (still extractive)
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
    # Build compact context from top chunks
    max_ctx = 12
    selected = ranked[:max_ctx]
    context_blocks: List[str] = []
    for c, _ in selected:
        context_blocks.append(
            f"[ID: {c.id}]\nArchivo: {c.source_name}\nUbicación: {c.location}\nContenido: {c.text}"
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
            # Fallback to responses API if available
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
                # responses.create returns output in a different structure
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


def render_app() -> None:
    st.set_page_config(page_title="H&R Hub - RAG", layout="centered")
    st.title("H&R Hub — Document-grounded RAG assistant")
    st.caption(
        "Answers strictly based on the documents in this project. "
        "Always include the 'Sources' section."
    )

    project_root = os.path.dirname(os.path.abspath(__file__))
    with st.spinner("Loading documents and building index…"):
        chunks = load_corpus(project_root)
        vectorizer, matrix = build_index(chunks)
    num_docs = len({c.source_path for c in chunks})
    num_chunks = len(chunks)
    st.write(f"Loaded documents: {num_docs} · Indexed chunks: {num_chunks}")

    if num_chunks == 0:
        st.warning(
            "No compatible documents (.pdf, .docx, .pptx) were found in the project. "
            "Add files to the existing folders and reload."
        )

    query = st.text_input("Type your question:", value="")
    top_k = st.slider("Number of chunks to consider", min_value=20, max_value=30, value=25)

    if st.button("Search", type="primary"):
        ranked = rank_chunks(query, vectorizer, matrix, chunks, top_k=top_k)
        # Try OpenAI generation respecting constraints
        ai_answer = call_openai_generate(query, ranked, max_sentences=5)
        if ai_answer is None or not ai_answer.strip():
            # Fallback to purely extractive compose_answer
            answer, sources = compose_answer(query, ranked)
            unavailable = answer.startswith("I cannot find information in the provided chunks to answer this.") or \
                answer.startswith("No se encuentra en la información disponible.")
            if not unavailable:
                sents = split_sentences(answer)
                if len(sents) > 6:
                    answer = " ".join(sents[:6]).strip()
            st.write(answer)
            st.write("\nSources:")
            if sources:
                for src in dedupe_preserve_order(sources, limit=10):
                    st.write(f"- {src}")
            else:
                st.write("- not specified")
        else:
            st.write(ai_answer)


if __name__ == "__main__":
    render_app()


