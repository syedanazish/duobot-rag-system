# DuoBot - Personal RAG Chatbot

**Live demo:** https://duobot-ragsystem-nlp.streamlit.app/

DuoBot is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about
**Syeda Nazish Fatima** and **Ayesha Khizar** (BSSE, UMT Lahore) using their personal documents.

Course: Natural Language Processing, instructor Sir Muhammad Ehsan.

## Architecture (follows the course diagram)

1. Encode documents: load, clean, chunk (500 chars, 80 overlap), embed with `all-MiniLM-L6-v2`
2. Index in a FAISS vector database (saved in `index/`)
3. Encode the user query with the same embedding model
4. Similarity search in FAISS
5. Retrieve the most similar chunks
6. Prompt = identity + rules + retrieved context + chat history + question (LangChain)
7. LLM (Groq, openai/gpt-oss-120b, free tier) generates the final answer

Extras: a small **agentic router** detects which person a question is about and filters retrieval by
metadata; a **query-condensing step** uses chat history to resolve follow-ups like "what about her?".
If an answer is not in the data, the bot says it does not have that information.

## Project structure

```
app.py            Streamlit UI
chat_cli.py       optional CLI
ingest.py         rebuild the FAISS index
rag/pipeline.py   loading, chunking, embeddings, retrieval, prompt, LLM
data/             personal documents (dataset)
index/            saved FAISS index (auto-created)
```

## Run locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your free key from https://console.groq.com in .env
python ingest.py
streamlit run app.py
```

If you edit files in `data/`, run `python ingest.py` again.

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, choose the repo, set main file to `app.py`.
3. In Advanced settings > Secrets add: `GROQ_API_KEY = "your_key"`
4. Deploy. Deployment link: **https://duobot-ragsystem-nlp.streamlit.app/**

## Team

- Syeda Nazish Fatima
- Ayesha Khizar
