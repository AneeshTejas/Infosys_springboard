import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/metadata.pkl"

def get_retriever(k=5):
    # Load metadata
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    documents = []
    index_to_docstore_id = {}

    for i, item in enumerate(metadata):
        doc_id = str(i)
        doc = Document(
            page_content=item["clean_text"],
            metadata={
                "aspect": item["aspect"],
                "sentiment": item["sentiment"],
                "confidence": item["confidence"],
                "source": item["source"],
            },
        )
        documents.append((doc_id, doc))
        index_to_docstore_id[i] = doc_id

    docstore = InMemoryDocstore({k: v for k, v in documents})

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

    return vectorstore.as_retriever(search_kwargs={"k": k})
