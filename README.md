# Nexus Info Internship Chatbots

## Overview

This repository contains two Streamlit-based AI chatbots, each designed for different use cases and levels of complexity:

### 1. Admission Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about admission-related documents (such as a university prospectus). It uses local PDF files, extracts and embeds their content, and provides context-aware answers using a large language model.

- **Features:**

  - Answers questions based only on the content of your uploaded PDFs.
  - Uses local HuggingFace embeddings (no OpenAI or Google API required).
  - Employs Groq LLM (Llama3-8b-8192) for generating responses.
  - Vector search is powered by FAISS for efficient retrieval.
  - Runs locally and privately.

- **Directory:** `admission-chatbot/`
- **Requirements:** See `admission-chatbot/requirements.txt`
- **How to run:**
  1. Place your PDFs in `admission-chatbot/docs/`.
  2. Add your Groq API key to `admission-chatbot/.env`.
  3. Install requirements and run:
     ```bash
     pip install -r requirements.txt
     streamlit run main.py
     ```

---

### 2. Simple Chatbot

A general-purpose conversational AI chatbot that uses Groq LLM for fast, interactive chat. It does not use document retrieval or embeddings, making it lightweight and easy to set up.

- **Features:**

  - General chat and Q&A with an LLM.
  - Customizable system prompt and model selection via the sidebar.
  - Maintains conversational memory for context-aware responses.
  - No document upload or retrieval—pure LLM chat.

- **Directory:** `simple-chatbot/`
- **Requirements:** See `simple-chatbot/requirements.txt`
- **How to run:**
  1. Add your Groq API key to `simple-chatbot/.env`.
  2. Install requirements and run:
     ```bash
     pip install -r requirements.txt
     streamlit run main.py
     ```

---

## Requirements

- Python 3.8+
- A Groq API key (get one from https://console.groq.com/)
- See each chatbot’s `requirements.txt` for specific dependencies.

## Troubleshooting

- **Missing Groq API key:** Ensure your `.env` file exists and contains the correct key.
- **Dependency errors:** Run `pip install -r requirements.txt` in the appropriate directory.
- **Admission Chatbot:** Make sure your PDFs are in the `docs/` directory.

## License

This project is for educational and demonstration purposes. Please check the licenses of all dependencies before using in production.
