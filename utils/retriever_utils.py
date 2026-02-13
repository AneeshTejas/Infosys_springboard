import os
import pickle
import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_PATH = r"D:\Infosys _Springboard\vector_db\faiss_store\index.faiss"
META_PATH = r"D:\Infosys _Springboard\vector_db\faiss_store\metadata.pkl"


def load_retriever(k=5):

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("FAISS store missing")

    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    docs = []
    index_to_docstore_id = {}

    for i, item in enumerate(metadata):
        doc_id = str(i)
        docs.append((doc_id, Document(
            page_content=item["clean_text"],
            metadata=item
        )))
        index_to_docstore_id[i] = doc_id

    docstore = InMemoryDocstore(dict(docs))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(INDEX_PATH)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )

    return vectorstore.as_retriever(search_kwargs={"k": k}), metadata
