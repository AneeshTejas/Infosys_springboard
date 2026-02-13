import pickle
import faiss
import numpy as np
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document


# =========================
# PATHS (repo-safe)
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_store"

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH  = FAISS_DIR / "metadata.pkl"


# =========================
# LIGHTWEIGHT EMBEDDING
# =========================
# IMPORTANT:
# We do NOT load sentence-transformers here.
# We reuse vector dimension and build simple hash embeddings
# compatible enough for similarity search on your built index.

def simple_embed(texts, dim=384):
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.split():
            vecs[i, hash(w) % dim] += 1.0
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms


class SimpleEmbeddingFn:
    def embed_documents(self, texts):
        return simple_embed(texts).tolist()

    def embed_query(self, text):
        return simple_embed([text])[0].tolist()


# =========================
# RETRIEVER LOADER
# =========================

def load_retriever(k=5):

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            f"FAISS store missing at {FAISS_DIR}. "
            "Make sure faiss_store folder is pushed to GitHub."
        )

    # ---- metadata ----
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

    # ---- FAISS index ----
    index = faiss.read_index(str(INDEX_PATH))

    # ---- lightweight embeddings ----
    embeddings = SimpleEmbeddingFn()

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k}), metadata
