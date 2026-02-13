import pickle
import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ============================================================
# PATHS
# ============================================================

INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/metadata.pkl"

# ============================================================
# LOAD METADATA
# ============================================================

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# ============================================================
# REBUILD DOCUMENT OBJECTS
# ============================================================

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

docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in documents})

# ============================================================
# EMBEDDINGS (MATCH BUILD PHASE)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# LOAD FAISS INDEX
# ============================================================

faiss_index = faiss.read_index(INDEX_PATH)

vectorstore = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

# ============================================================
# TEST RETRIEVAL
# ============================================================

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

query = "refund not released"
results = retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"Rank {i}")
    print("Text:", doc.page_content)
    print("Meta:", doc.metadata)
    print("-" * 60)
