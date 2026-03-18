# 📄 PDF Chat Bot

An AI-powered chatbot that allows users to upload PDF documents and ask natural language questions to retrieve contextual answers — built using RAG (Retrieval-Augmented Generation) architecture.

---

## 🌐 Live Demo

👉 [pdf-chat-bot-arman.streamlit.app](https://pdf-chat-bot-arman.streamlit.app/)

---

## 📸 Snapshots

**Home — Ready to query**

![PDF Chat Bot Home](snapshots/snapshot1.png)

**Response — Answering from document**

![PDF Chat Bot Response](snapshots/snapshot.png)

---

## 🚀 Project Overview

### 📌 Problem Statement

Understanding information from large PDF documents is time-consuming and inefficient. Users often need to manually search through long documents to find relevant answers.

### 🎯 Solution

Developed an AI-powered chatbot that allows users to upload PDF documents and ask natural language questions to retrieve contextual answers.

### 🧠 Core Idea

Convert PDF text into vector embeddings and use semantic search with a Generative AI model to answer user queries accurately.

---

## 👨‍💻 My Role & Contribution

- Designed full backend logic
- Implemented document parsing pipeline
- Built vector similarity search
- Integrated OpenRouter LLM with LangChain
- Developed Streamlit UI
- Migrated from Gemini API to OpenRouter + HuggingFace embeddings

---

## 🏗️ System Architecture

### 🔄 Workflow Pipeline

```
Upload PDF → Extract Text → Chunk Text → Generate Embeddings (HuggingFace) → Store in FAISS → User Query → Similarity Search → LLM Generates Answer (OpenRouter)
```

---

## 🧩 Tech Stack & Why I Chose It

| Category | Technology | Reason |
|---|---|---|
| UI | Streamlit | Rapid development, minimal frontend overhead |
| PDF Parsing | PyPDF2 | Simple and reliable text extraction |
| Text Splitting | LangChain | Handles chunking and overlap efficiently |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) | Free, runs locally, no API dependency |
| Vector Store | FAISS | Fast similarity search, handles large embeddings |
| LLM | OpenRouter (`arcee-ai/trinity-large-preview:free`) | Free tier, never expires |
| Environment | python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
PDF_Chat_Bot/
│
├── main.py               # Main application file
├── requirements.txt      # Python dependencies
├── snapshots/            # App screenshots
├── .env                  # API keys (not pushed to GitHub)
├── .gitignore            # Files ignored by git
└── README.md             # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arman1263/pdf-chat-bot.git
cd pdf-chat-bot
```

### 2. Create and activate virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
source myenv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your free API key at [openrouter.ai](https://openrouter.ai)

### 5. Run the application

```bash
streamlit run main.py
```

---

## 📖 How to Use

1. Open the app in your browser
2. Upload one or more PDF files using the sidebar
3. Click **Process Documents** to index the PDFs
4. Type your question in the input box
5. Click **Submit Question** to get an answer

---

## ⚙️ Implementation Breakdown

### 📥 Step 1 — PDF Upload & Text Extraction

- Users upload multiple PDFs via Streamlit sidebar
- Text extracted using PyPDF2
- Combined text stored for processing

### ✂️ Step 2 — Text Chunking

LLMs have token limits — large documents must be split into smaller sections.

- Used LangChain `RecursiveCharacterTextSplitter`
- Chunk size: 10,000 characters with 1,000 character overlap
- Overlap improves context retention across chunks

### 🧠 Step 3 — Embedding Generation

- Text chunks converted into vector embeddings
- Used HuggingFace `all-MiniLM-L6-v2` model (runs locally)
- Embeddings stored in FAISS index saved to disk

### 🔎 Step 4 — Similarity Search

- User query converted into embedding using same HuggingFace model
- FAISS retrieves most semantically relevant chunks
- Top results passed as context to the LLM

### 💬 Step 5 — Response Generation

- LLM (via OpenRouter) processes query + retrieved context
- Generates final answer using LangChain LCEL pipeline
- Answer displayed in Streamlit UI

---

## 🔥 Major Challenges Faced

### 1. Handling Large PDF Files

**Problem:** Large PDFs caused memory and performance issues.  
**Solution:** Implemented chunking with overlap.  
**Learning:** Token limits and chunk strategy are critical in LLM applications.

### 2. FAISS Index Dimension Mismatch

**Problem:** After migrating from Gemini to HuggingFace embeddings, FAISS index threw dimension mismatch errors (768 vs 384 dimensions).  
**Solution:** Deleted old FAISS index and rebuilt it with new embeddings.  
**Learning:** Embedding models must stay consistent between index creation and query time.

### 3. LangChain Deprecation Issues

**Problem:** Several LangChain modules (`load_qa_chain`, `langchain.text_splitter`, `langchain.vectorstores`) were deprecated or moved in newer versions.  
**Solution:** Migrated to LCEL pipeline using `langchain_core` and updated all imports.  
**Learning:** LangChain evolves fast — always check module paths against the current version.

### 4. Maintaining Context Accuracy

**Problem:** Model sometimes generated answers unrelated to the document.  
**Solution:** Improved retrieval quality and chunk overlap strategy.  
**Learning:** Retrieval quality impacts LLM output more than model power.

---

## 📊 Performance Considerations

- Chunk size and overlap tuning for better context
- FAISS local indexing for fast retrieval
- HuggingFace embeddings run locally — no network latency
- Lazy loading of documents on demand

---

## 🧪 Testing

- Tested with research papers
- Tested with structured and unstructured PDFs
- Verified contextual answer correctness against known document content

---

## 🔐 Security

- API keys managed using `.env` and `python-dotenv`
- `.env` excluded from version control via `.gitignore`
- FAISS index loading restricted with `allow_dangerous_deserialization` flag

---

## 🌱 Future Improvements

- Chat history memory across sessions
- Highlight answer sources inside PDF
- Multi-document tagging and filtering
- Deployment using Docker
- Switch to scalable vector DB (Pinecone / Chroma)

---

## 💡 Key Learnings

- End-to-end RAG pipeline building
- Vector databases and embeddings
- Prompt engineering basics
- LangChain LCEL pipeline design
- Handling token limits in real systems
- API migration and dependency management

---

## ⭐ STAR Interview Story

**Situation:** Users struggle to quickly extract insights from lengthy PDF documents.

**Task:** Build an AI system that answers natural language queries from PDF content.

**Action:**
- Built document ingestion and chunking pipeline
- Implemented FAISS vector similarity search
- Integrated LLM via OpenRouter with LangChain LCEL
- Developed Streamlit UI
- Migrated from Gemini to OpenRouter + HuggingFace when API quota ran out

**Result:** Successfully built an AI chatbot capable of delivering contextual answers from PDFs, demonstrating practical skills in RAG architecture, API integration, and real-world problem solving.

---

## 📦 Requirements

```
streamlit
PyPDF2
langchain
langchain-openai
langchain-huggingface
langchain-community
langchain-core
langchain-text-splitters
sentence-transformers
faiss-cpu
python-dotenv
```

---

## 👤 Author

**Arman Amir Shikalgar**  
B.Tech in Artificial Intelligence & Data Science  
[GitHub](https://github.com/Arman1263)
