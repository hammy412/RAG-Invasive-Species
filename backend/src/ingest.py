import os
import fitz  # PyMuPDF for PDF + OCR fallback
import pytesseract
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from trafilatura import extract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

RAW_DIR = "../data_raw/"
OUT_PATH = "../data_processed/chunks.json"


# PDF Extraction 

def extract_pdf(path):
    doc = fitz.open(path)
    pages = []

    for page in doc:
        text = page.get_text("text")

        if text.strip():
            pages.append(text)
            continue

        # Fallback OCR when no text is extracted
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        pages.append(text)

    return "\n".join(pages)



# HTML Extraction

def extract_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    clean = extract(html, include_tables=True)
    if clean:
        return clean

    # fallback to BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# Load ALL documents recursively

def load_documents():
    docs = []

    for root, _, files in os.walk(RAW_DIR):
        for filename in files:
            path = os.path.join(root, filename)
            ext = filename.lower()

            try:
                if ext.endswith(".pdf"):
                    text = extract_pdf(path)
                elif ext.endswith(".html") or ext.endswith(".htm"):
                    text = extract_html(path)
                elif ext.endswith(".txt") or ext.endswith(".md"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                else:
                    continue  # unsupported file

                if text and text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": path,
                            "filename": filename,
                            "category": root.replace(RAW_DIR, "")
                        }
                    ))

            except Exception as e:
                print(f"❌ Failed to process {path}: {e}")

    print(f"Loaded {len(docs)} documents.")
    return docs



# Chunking

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)



# MAIN
if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    import json
    os.makedirs("data_processed", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": c.page_content, "meta": c.metadata} for c in chunks],
            f,
            indent=2
        )

    print("✅ Saved processed chunks →", OUT_PATH)