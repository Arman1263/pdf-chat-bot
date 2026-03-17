# 📄 PDF Chat Bot

An AI-powered application that lets you upload PDF files and ask questions about their content using natural language. Built with Streamlit, LangChain, and OpenRouter API.

---

## 🚀 Features

- Upload single or multiple PDF files
- Ask questions about the content of uploaded PDFs
- Accurate answers powered by a free LLM via OpenRouter
- Local vector search using FAISS for fast document retrieval
- Clean and simple UI built with Streamlit

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| LLM API | OpenRouter (`arcee-ai/trinity-large-preview:free`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| PDF Parsing | PyPDF2 |
| Framework | LangChain |
| Environment | Python 3.10+ |

---

## 📁 Project Structure

```
PDF_Chat_Bot/
│
├── main.py               # Main application file
├── requirements.txt      # Python dependencies
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
3. Click **Submit & Process** to index the PDFs
4. Type your question in the input box
5. Click **Submit Question** to get an answer

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

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key |

---

## 👤 Author

**Arman Amir Shikalgar**  
B.Tech in Artificial Intelligence & Data Science  
[GitHub](https://github.com/Arman1263)
