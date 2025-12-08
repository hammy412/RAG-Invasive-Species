from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------------------
# INIT ONCE AT STARTUP
# ---------------------

PERSIST_DIR = "../chroma_db/"

app = FastAPI(title="Invasive Species QA API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Embedding model must match database
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3", temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about invasive species.

Use ONLY the provided context. Do NOT make up facts.

Question: {question}

Context:
{context}

Answer:
""")


# ---------------------
# REQUEST / RESPONSE MODELS
# ---------------------

class AskRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]


# ---------------------
# API ENDPOINT
# ---------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question

    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)

    # 2. Build LLM context
    context_text = "\n\n".join(d.page_content for d in docs)

    # 3. Format the final prompt
    final_prompt = prompt.invoke({
        "question": question,
        "context": context_text
    })

    # 4. Run LLM
    result = llm.invoke(final_prompt)
    answer = result.content

    # 5. Prepare chunks for API response
    chunks_output = []
    for d in docs:
        chunks_output.append(
            RetrievedChunk(
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                chunk_id=d.metadata.get("chunk")
            )
        )

    return AskResponse(
        answer=answer,
        retrieved_chunks=chunks_output
    )
