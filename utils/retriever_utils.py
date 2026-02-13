from pathlib import Path
import pickle
import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ----------- Robust Project Paths (Cloud + Local Safe) -----------

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_store"

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "metadata.pkl"


# ----------- Retriever Loader -----------

def load_retriever(k: int = 5):

    # ---- Existence check (Path-safe) ----
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            f"FAISS store missing.\n"
            f"Expected index at: {INDEX_PATH}\n"
            f"Expected metadata at: {META_PATH}"
        )

    # ---- Load metadata ----
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # ---- Build LangChain docstore ----
    docs = {}
    index_to_docstore_id = {}

    for i, item in enumerate(metadata):
        doc_id = str(i)

        docs[doc_id] = Document(
            page_content=item["clean_text"],
            metadata=item
        )

        index_to_docstore_id[i] = doc_id

    docstore = InMemoryDocstore(docs)

    # ---- Embedding model ----
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    )

    # ---- Load FAISS index (string path required) ----
    index = faiss.read_index(str(INDEX_PATH))

    # ---- Build vectorstore ----
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever, metadata
