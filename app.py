import hashlib
import os
import time

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PDF = "handbook.pdf"  # put your college handbook PDF next to app.py

st.set_page_config(page_title="Campus Helper", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------
# Colorful styling
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Animated rainbow-ish gradient background */
    .stApp {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #a18cd1,
                                    #fbc2eb, #84fab0, #8fd3f4, #ffecd2);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Transparent top bar */
    header[data-testid="stHeader"] { background: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6a11cb 0%, #2575fc 100%);
    }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }

    /* Title banner */
    .title-card {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 18px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        text-align: center;
    }
    .title-card h1 {
        margin: 0;
        background: linear-gradient(90deg, #ff512f, #dd2476, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
    }
    .title-card p { margin: 4px 0 0 0; color: #333; }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
    }
    div[data-testid="stChatMessage"] * { color: #1f1f2e; }

    /* Chat input */
    div[data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
        <h1>🎓 Campus Helper Chatbot</h1>
        <p>Ask anything about your college handbook – rules, fees, attendance, exams and more!</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()


def read_pdf(source):
    """Return a list of (page_number, text) from a path or uploaded file."""
    reader = PdfReader(source)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, size=900, overlap=150):
    chunks, metas = [], []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(piece)
                metas.append({"page": page_no})
            start += size - overlap
    return chunks, metas


def build_index(pdf_bytes_or_path, key):
    pages = read_pdf(pdf_bytes_or_path)
    if not pages:
        return None, 0
        st.session_state.full_text = "\n\n".join(f"[Page {p}] {t}" for p, t in pages)
    chunks, metas = chunk_pages(pages)
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    client = get_chroma_client()
    name = f"handbook_{key}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=metas,
    )
    return name, len(chunks)


def retrieve(question, k=4):
    client = get_chroma_client()
    collection = client.get_collection(st.session_state.collection)
    q_emb = load_embedder().encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in hits)
    prompt = (
        "You are Campus Helper, a friendly assistant for college students. "
        "Answer ONLY using the handbook context below. If the answer is not "
        "in the context, say you couldn't find it in the handbook and suggest "
        "contacting the college office. Keep the answer clear and short. "
        "Always format the answer as bullet points (one point per line, "
        "starting with '* '), never as a paragraph. Start with one short "
        "line like 'Based on the handbook (Page X):' and then the bullets. "
        "Mention page numbers when useful.\n\n"
        f"Handbook context:\n{context}\n\n"
        f"Student question: {question}\nAnswer:"
    )
    pages = sorted({m["page"] for _, m in hits})
    client = genai.Client(api_key=API_KEY)
    last_error = None
    for model in GEN_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text, pages
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                if "404" in msg:
                    break
                raise
    raise last_error
    
    client = genai.Client(api_key=API_KEY)
    last_error = None
    for model in GEN_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                pages = sorted({m["page"] for _, m in hits})
                return response.text, pages
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                    time.sleep(2 * (attempt + 1))  # wait 2s, 4s, 6s then retry
                    continue
                if "404" in msg:
                    break  # model not available, try the next one
                raise
        raise RuntimeError(f"All models failed. Last error: {last_error}")

# ---------------------------------------------------------------
# Sidebar: choose the PDF
# ---------------------------------------------------------------
with st.sidebar:
    st.header("📚 Handbook")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    st.caption(f"Or place `{DEFAULT_PDF}` in the same folder as app.py.")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if not API_KEY:
        st.error("GEMINI_API_KEY not found. Add it to a .env file.")

# ---------------------------------------------------------------
# Build the index (once per PDF)
# ---------------------------------------------------------------
source, key = None, None
if uploaded is not None:
    data = uploaded.getvalue()
    source, key = uploaded, hashlib.md5(data).hexdigest()[:10]
elif os.path.exists(DEFAULT_PDF):
    with open(DEFAULT_PDF, "rb") as f:
        key = hashlib.md5(f.read()).hexdigest()[:10]
    source = DEFAULT_PDF

if source is None:
    st.info("👈 Upload your college handbook PDF in the sidebar to get started.")
    st.stop()

if st.session_state.get("pdf_key") != key:
    with st.spinner("Reading and indexing the handbook..."):
        name, n = build_index(source, key)
    if name is None:
        st.error("Couldn't extract text from this PDF (it may be scanned images).")
        st.stop()
    st.session_state.pdf_key = key
    st.session_state.collection = name
    st.session_state.messages = []
    with st.sidebar:
        st.success(f"Indexed {n} chunks ✅")

# ---------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hi! 👋 I'm Campus Helper. What would you like to know?"}
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about attendance, fees, exams, hostel..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                text, pages = answer(question)
                reply = f"{text}\n\n📄 *Sources: page {', '.join(map(str, pages))}*"
            except Exception as e:
                reply = f"⚠️ Something went wrong: `{e}`"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})