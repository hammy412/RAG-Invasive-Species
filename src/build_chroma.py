import json
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHUNKS_PATH = "data_processed/chunks.json"
PERSIST_DIR = "chroma_db/"

def load_chunks():
    with open(CHUNKS_PATH, "r") as f:
        data = json.load(f)

    # Recreate LangChain Document objects
    from langchain_core.documents import Document
    docs = [
        Document(page_content=item["text"], metadata=item["meta"])
        for item in data
    ]
    return docs


def build_chroma(docs):
    # You can swap embeddings later for smaller/faster
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectorstore.persist()
    print("Chroma DB saved to:", PERSIST_DIR)


if __name__ == "__main__":
    docs = load_chunks()
    print(f"Loaded {len(docs)} chunks.")
    build_chroma(docs)
