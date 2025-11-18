import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

PERSIST_DIR = "chroma_db/"

# Embedding model must match DB
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3", temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about invasive species.

Use ONLY the provided context.

Question: {question}

Context:
{context}

Answer:
""")


def ask(question: str):
    # 1. Retrieve source chunks
    docs = retriever.invoke(question)

    # 2. Build LLM context
    context_text = "\n\n".join(d.page_content for d in docs)

    # 3. Run LLM
    final_prompt = prompt.invoke({
        "question": question,
        "context": context_text
    })

    result = llm.invoke(final_prompt)

    # 4. Print answer
    print("\n🧠 Answer:\n")
    print(result.content)

    # 5. Print the EXACT chunks used
    print("\n📌 EXACT Chunks Used (in retrieval order):\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk #{i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        if "chunk" in doc.metadata:
            print(f"Chunk ID: {doc.metadata['chunk']}")
        print("\n" + doc.page_content + "\n")



if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        ask(q)
