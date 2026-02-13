import pickle
import faiss
import numpy as np
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_store"

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH  = FAISS_DIR / "metadata.pkl"


# =========================
# LIGHTWEIGHT EMBEDDINGS (LangChain compatible)
# =========================

class SimpleHashEmbeddings(Embeddings):
    """
    Torch-free, sentence-transformer-free embeddings
    Works on Streamlit Cloud
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str):
        vec = np.zeros(self.dim, dtype="float32")
        for w in text.split():
            vec[hash(w) % self.dim] += 1.0

        norm = np.linalg.norm(vec) + 1e-9
        return (vec / norm).tolist()

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]


# =========================
# RETRIEVER
# =========================

def load_retriever(k=5):

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            f"FAISS store missing at {FAISS_DIR}. "
            "Ensure faiss_store folder is committed to repo."
        )

    # ---- load metadata ----
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

    # ---- load FAISS ----
    index = faiss.read_index(str(INDEX_PATH))

    # ---- embeddings ----
    embeddings = SimpleHashEmbeddings(dim=index.d)

    vectorstore = FAISS(
        embedding_function=embeddings,   # ✅ correct type now
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k}), metadata
