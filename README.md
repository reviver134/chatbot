# 📚 RAG Chatbot with LangChain + Ollama

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain. It answers questions based on the content of a local text file (`text.txt`) by retrieving relevant document chunks and using an Ollama LLM to generate responses.

---

## Features

- Loads and splits text documents into chunks for efficient retrieval.
- Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) for vector similarity search with Chroma vector store.
- Retrieves top relevant document chunks for the query.
- Generates answers with the Ollama LLM (`llama2` model) using a custom prompt template.
- Displays the prompt sent to the model, the answer, and source documents in a user-friendly Streamlit interface.

---

## Requirements

- Python 3.8+
- Streamlit
- LangChain and related community packages (`langchain_community`, `langchain_ollama`)
- Chroma vector store
- Ollama LLM setup (ensure `llama2` model is available locally via Ollama)
- PyTorch (fix handled automatically in code)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
