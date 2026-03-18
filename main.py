import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat Bot",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #050a12 !important;
    color: #c8d8e8 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,180,255,0.08) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(0,180,255,0.02) 40px, rgba(0,180,255,0.02) 41px),
        repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(0,180,255,0.02) 40px, rgba(0,180,255,0.02) 41px);
    background-attachment: fixed;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 760px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,180,255,0.15) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #4a7a9a !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #00b4ff !important;
    border-bottom: 2px solid #00b4ff !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,180,255,0.04) !important;
    border: 1px dashed rgba(0,180,255,0.25) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,180,255,0.5) !important;
}
[data-testid="stFileUploader"] label {
    color: #7aa8c8 !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid rgba(0,180,255,0.4) !important;
    color: #00b4ff !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(0,180,255,0.1) !important;
    border-color: #00b4ff !important;
    box-shadow: 0 0 18px rgba(0,180,255,0.2) !important;
    color: #fff !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(0,180,255,0.04) !important;
    border: 1px solid rgba(0,180,255,0.2) !important;
    border-radius: 6px !important;
    color: #c8d8e8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #00b4ff !important;
    box-shadow: 0 0 0 2px rgba(0,180,255,0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #3a5a7a !important;
}
[data-testid="stTextInput"] label {
    color: #7aa8c8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050a12; }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,180,255,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 2rem;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.4rem;">
        <div style="
            width:36px; height:36px;
            background: linear-gradient(135deg, #00b4ff22, #0050ff22);
            border: 1px solid rgba(0,180,255,0.4);
            border-radius: 8px;
            display:flex; align-items:center; justify-content:center;
            font-size:1.1rem;">⬡</div>
        <div>
            <div style="
                font-family:'Space Mono',monospace;
                font-size:0.65rem;
                letter-spacing:0.3em;
                text-transform:uppercase;
                color:#00b4ff;
                margin-bottom:2px;">
                RAG · FAISS · OpenRouter
            </div>
            <h1 style="
                font-family:'Syne',sans-serif;
                font-size:1.75rem;
                font-weight:800;
                color:#e8f4ff;
                margin:0;
                line-height:1;
                letter-spacing:-0.02em;">
                PDF Chat Bot
            </h1>
        </div>
    </div>
    <div style="
        height:1px;
        background: linear-gradient(90deg, rgba(0,180,255,0.4), rgba(0,80,255,0.2), transparent);
        margin-top:1rem;">
    </div>
    <p style="
        color:#4a7a9a;
        font-size:0.82rem;
        margin-top:0.6rem;
        letter-spacing:0.02em;">
        Upload PDFs · Ask questions · Get precise answers from your documents
    </p>
</div>
""", unsafe_allow_html=True)


# ── Core functions ─────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(model):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide the wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    model = ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        temperature=0.3,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    chain = get_conversational_chain(model)
    response = chain.invoke({"context": context, "question": user_question})

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(0,180,255,0.05), rgba(0,50,100,0.08));
        border: 1px solid rgba(0,180,255,0.18);
        border-left: 3px solid #00b4ff;
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        font-family: 'Syne', sans-serif;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #c8d8e8;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    ">
        <div style="
            font-family:'Space Mono',monospace;
            font-size:0.6rem;
            letter-spacing:0.25em;
            text-transform:uppercase;
            color:#00b4ff;
            margin-bottom:0.7rem;
            opacity:0.8;">
            ◈ Response
        </div>
        {response}
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["◈  Ask", "⬡  Upload PDF"])

# ── Tab 1: Ask ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    def clear_input():
        st.session_state["user_question"] = ""

    st.markdown("""
    <div style="
        font-family:'Space Mono',monospace;
        font-size:0.65rem;
        letter-spacing:0.2em;
        text-transform:uppercase;
        color:#2a5a7a;
        margin-bottom:0.6rem;">
        ◈ Ask a question
    </div>
    """, unsafe_allow_html=True)

    st.text_input(
        "Query",
        placeholder="What is this document about?",
        key="user_question",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("⬡  Submit Question"):
            if st.session_state.user_question:
                with st.spinner("Retrieving answer..."):
                    user_input(st.session_state.user_question)
            else:
                st.warning("Enter a question first.")
    with col2:
        st.button("✕  Clear", on_click=clear_input)

# ── Tab 2: Upload ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        font-family:'Space Mono',monospace;
        font-size:0.65rem;
        letter-spacing:0.2em;
        text-transform:uppercase;
        color:#2a5a7a;
        margin-bottom:0.6rem;">
        ⬡ Upload & Process
    </div>
    """, unsafe_allow_html=True)

    pdf_docs = st.file_uploader(
        "Drop PDF files here",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="visible"
    )

    if pdf_docs:
        st.markdown(f"""
        <div style="
            font-family:'Space Mono',monospace;
            font-size:0.68rem;
            color:#4a9a7a;
            letter-spacing:0.08em;
            margin: 0.5rem 0 1rem;
            padding: 0.4rem 0.8rem;
            background: rgba(0,180,100,0.06);
            border: 1px solid rgba(0,180,100,0.15);
            border-radius:4px;">
            ✓ {len(pdf_docs)} file(s) ready
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬡  Process Documents"):
        if pdf_docs:
            with st.spinner("Indexing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done — switch to Ask tab to query your PDF.")
        else:
            st.warning("Upload at least one PDF first.")
