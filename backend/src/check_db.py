from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_db/"

# Use SAME embedding model used during build
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

print("Number of vectors in Chroma DB:", db._collection.count())

items = db._collection.get(include=['embeddings'])

print("Number of embeddings:", len(items["embeddings"]))
print("First embedding dimension:", len(items["embeddings"][0]))
print("First 10 values of the first embedding:", items["embeddings"][0][:10])
