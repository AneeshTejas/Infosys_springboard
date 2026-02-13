from pathlib import Path
import os
import pickle
import faiss

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# ENV — Groq API
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")


# =========================
# PATHS — Cloud Safe
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_store"

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH  = FAISS_DIR / "metadata.pkl"


# =========================
# RAG CHAIN
# =========================

def build_rag_chain():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
You are analyzing Swiggy customer reviews.

Answer ONLY from provided context.

Question:
{question}

Context:
{context}

Return:
- Direct answer
- Pattern
- Dominant aspect
- Sentiment trend
""")

    return prompt | llm | StrOutputParser()


# =========================
# FAISS RETRIEVER (NO LLM)
# =========================

def load_faiss_retriever_only(k=5):

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("FAISS store missing")

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    docs = {}
    index_to_docstore_id = {}

    for i, item in enumerate(metadata):
        did = str(i)
        docs[did] = Document(
            page_content=item["clean_text"],
            metadata=item
        )
        index_to_docstore_id[i] = did

    docstore = InMemoryDocstore(docs)

    # ---- SAFE EMBEDDINGS ----
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    index = faiss.read_index(str(INDEX_PATH))

    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vs.as_retriever(search_kwargs={"k": k})
