import os
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, BSHTMLLoader
from trafilatura import extract
from langchain_core.documents import Document


RAW_DIR = "data_raw/"
OUT_PATH = "data_processed/chunks.json"

def load_documents():
    docs = []

    # Load PDFs
    for f in os.listdir(RAW_DIR):
        if f.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(RAW_DIR, f))
            docs.extend(loader.load())
        elif f.lower().endswith(".html"):
            docs.extend(load_clean_html(os.path.join(RAW_DIR, f)))
        elif f.lower().endswith(".txt") or f.lower().endswith(".md"):
            loader = TextLoader(os.path.join(RAW_DIR, f), encoding="utf-8")
            docs.extend(loader.load())


    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

def load_clean_html(path):
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

        text = extract(html, include_comments=False, include_tables=True)
        if not text:
            return []

        return [Document(page_content=text, metadata={"source": path})]


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    # Save to JSON
    import json
    with open(OUT_PATH, "w") as f:
        json.dump([{"text": c.page_content, "meta": c.metadata} for c in chunks], f, indent=2)

    print("Saved chunks →", OUT_PATH)
