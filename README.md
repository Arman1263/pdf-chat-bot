# PDF-Chat-Bot
Upload the PDF Files and ask questions

## To see Preview
https://pdf-chat-bot-arman.streamlit.app/

# Chat with PDF using Gemini 💁

This project is a **Streamlit-based application** that allows users to interactively chat with PDF documents using Google's Gemini AI. It enables users to upload PDF files, process them, and ask questions based on the content of the files.

---

## Features
- 📄 Upload and process multiple PDF files.
- 💬 Ask questions based on the content of the PDFs.
- ⚡ Fast and detailed answers using **Google Gemini AI**.
- 🧠 Leverages **FAISS** for efficient similarity search and text embeddings.
- 🔥 Supports chunked text processing for large PDFs.

---

## Tech Stack
- **Python** 🐍
- **Streamlit** for UI
- **PyPDF2** for PDF reading
- **LangChain** for text processing and AI integration
- **FAISS** for vector similarity search
- **Google Gemini AI** (Generative AI model)
- **dotenv** for managing environment variables

---

## Prerequisites
1. Python 3.9 or later
2. A valid [Google API Key](https://developers.generativeai.google/).
3. The following Python libraries installed:
   - `streamlit`
   - `PyPDF2`
   - `langchain`
   - `langchain_google_genai`
   - `google-generativeai`
   - `faiss-cpu`
   - `python-dotenv`

---

## Known Issues
1. Pickle Deserialization Warning: Ensure FAISS index files are from trusted sources. If encountering a ValueError, delete the faiss_index folder and reprocess the PDFs.
2. Large PDFs: Processing very large PDFs may take additional time.

![PDF Chat Bot](https://github.com/Arman1263/pdf-chat-bot/blob/b085a67deffffa3e052493ff5aa4afcd94e7ea58/7.png)

## So called Arman😇
