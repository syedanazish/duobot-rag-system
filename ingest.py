"""Run once (or after editing files in data/) to rebuild the FAISS index."""
from rag.pipeline import build_vectorstore, chunk_documents, load_documents

docs = load_documents()
print(f"Loaded {len(docs)} documents, {len(chunk_documents(docs))} chunks")
build_vectorstore(force_rebuild=True)
print("Index saved to index/")
