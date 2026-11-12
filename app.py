import os
import json
import time
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.5rem; max-width: 760px;}
    h1 {letter-spacing: -0.02em;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.25);}
    .empty-box {
        padding: 1.5rem; border-radius: 14px; margin-top: 1rem;
        border: 1.5px dashed rgba(59, 63, 216, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def find_relevant(question, chunks, vectors, embedder, k=4):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    best = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in best]


def get_key():
    # On Streamlit Cloud the key comes from Secrets; on your laptop from the terminal
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ["GEMINI_API_KEY"]


def generate(prompt, json_mode=False):
    client = genai.Client(api_key=get_key())
    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for attempt in range(4):
        try:
            reply = client.models.generate_content(model=MODEL, contents=prompt, config=config)
            return reply.text
        except Exception as e:
            busy = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if busy and attempt < 3:
                time.sleep(2 * (attempt + 1))  # wait 2s, 4s, 6s, then try again
            else:
                raise


def ask_llm(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a study assistant. Answer the question using ONLY the notes below. "
        "If the answer is not in the notes, say so.\n\n"
        f"NOTES:\n{context}\n\nQUESTION: {question}"
    )
    return generate(prompt)


def make_quiz(topic, n, chunks, vectors, embedder):
    # Pick the notes the quiz will be based on: matching the topic, or a random sample
    if topic.strip():
        picked = find_relevant(topic, chunks, vectors, embedder, k=8)
    else:
        idx = np.random.choice(len(chunks), size=min(8, len(chunks)), replace=False)
        picked = [chunks[i] for i in idx]
    context = "\n\n---\n\n".join(picked)
    prompt = (
        f"Create {n} multiple-choice questions based ONLY on the notes below. "
        "Return ONLY a JSON list. Each item must have exactly these keys: "
        '"question" (string), "options" (list of exactly 4 strings), '
        '"answer_index" (integer 0 to 3, the position of the correct option), '
        '"explanation" (one short sentence). Make the wrong options plausible.\n\n'
        f"NOTES:\n{context}"
    )
    text = generate(prompt, json_mode=True).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("questions", [])
    quiz = []
    for item in data:
        ok = (
            isinstance(item, dict)
            and isinstance(item.get("options"), list)
            and len(item["options"]) == 4
            and isinstance(item.get("answer_index"), int)
            and 0 <= item["answer_index"] < 4
            and "question" in item
        )
        if ok:
            quiz.append(item)
    return quiz


# ---------- Sidebar: upload, process, mode ----------
with st.sidebar:
    st.header("Your notes")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Process notes", type="primary", use_container_width=True):
        embedder = load_embedder()
        chunks = []
        for f in files:
            chunks += chunk_text(read_pdf(f))
        with st.spinner("Reading your notes..."):
            vectors = embedder.encode(chunks, normalize_embeddings=True)
        st.session_state["chunks"] = chunks
        st.session_state["vectors"] = vectors
        st.session_state["messages"] = []
        st.session_state.pop("quiz", None)
    if "chunks" in st.session_state:
        st.success(f"Ready: {len(st.session_state['chunks'])} pieces of notes loaded.")
        mode = st.radio("What do you want to do?", ["Chat", "Quiz"])
        if mode == "Chat" and st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
    else:
        mode = "Chat"

# ---------- Main area ----------
st.title("📚 Study Assistant")
st.caption("Ask questions about your own notes, or test yourself with a quiz.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chunks" not in st.session_state:
    st.markdown(
        '<div class="empty-box"><b>Get started</b><br>'
        "Upload a PDF in the sidebar, click <b>Process notes</b>, "
        "then ask questions or take a quiz.</div>",
        unsafe_allow_html=True,
    )

elif mode == "Chat":
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources used"):
                    for s in m["sources"]:
                        st.caption(s)

    question = st.chat_input("Ask a question about your notes")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        embedder = load_embedder()
        top = find_relevant(question, st.session_state["chunks"], st.session_state["vectors"], embedder)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = ask_llm(question, top)
                st.markdown(answer)
                with st.expander("Sources used"):
                    for s in top:
                        st.caption(s)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "sources": top}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.subheader("Quiz me")
    topic = st.text_input("Topic (optional)", placeholder="e.g. BFS, transformers. Leave empty for a mix.")
    n = st.slider("Number of questions", 3, 10, 5)

    if st.button("Generate quiz", type="primary"):
        try:
            with st.spinner("Writing your questions..."):
                quiz = make_quiz(
                    topic, n,
                    st.session_state["chunks"], st.session_state["vectors"], load_embedder(),
                )
            if not quiz:
                raise ValueError("No valid questions came back. Please try again.")
            st.session_state["quiz"] = quiz
            st.session_state["quiz_id"] = st.session_state.get("quiz_id", 0) + 1
            st.session_state["submitted"] = False
        except Exception as e:
            st.error(f"Could not make the quiz: {e}")

    quiz = st.session_state.get("quiz")
    if quiz:
        qid = st.session_state["quiz_id"]
        submitted = st.session_state.get("submitted", False)
        st.divider()
        for i, item in enumerate(quiz):
            st.markdown(f"**Q{i + 1}. {item['question']}**")
            choice = st.radio(
                "Choose one", item["options"], index=None,
                key=f"q{qid}_{i}", disabled=submitted, label_visibility="collapsed",
            )
            if submitted:
                correct = item["options"][item["answer_index"]]
                note = item.get("explanation", "")
                if choice == correct:
                    st.success(f"Correct! {note}")
                else:
                    st.error(f"The right answer is: {correct}. {note}")

        if not submitted:
            if st.button("Submit answers"):
                st.session_state["submitted"] = True
                st.rerun()
        else:
            score = sum(
                1 for i, item in enumerate(quiz)
                if st.session_state.get(f"q{qid}_{i}") == item["options"][item["answer_index"]]
            )
            st.metric("Your score", f"{score} / {len(quiz)}")
            st.caption("Use Generate quiz above to get a new set of questions.")
