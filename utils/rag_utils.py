from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
import pickle
import faiss
import os
from pathlib import Path
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")



def build_rag_chain():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # stable + fast
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

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_store" / "index.faiss"
META_PATH  = BASE_DIR / "faiss_store" / "metadata.pkl"


def load_faiss_retriever_only(k=5):

    with open(META_PATH,"rb") as f:
        metadata = pickle.load(f)

    docs = {}
    index_to_id = {}

    for i,item in enumerate(metadata):
        did = str(i)
        docs[did] = Document(
            page_content=item["clean_text"],
            metadata=item
        )
        index_to_id[i] = did

    docstore = InMemoryDocstore(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(INDEX_PATH)

    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_id,
    )

    return vs.as_retriever(search_kwargs={"k":k})
