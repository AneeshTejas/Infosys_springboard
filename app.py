import streamlit as st
from utils.analytics_utils import load_dataframe

st.set_page_config(
    page_title="Swiggy Review Intelligence Platform",
    layout="wide"
)

# -----------------------------
# Rename Sidebar "app" → "Dashboard"
# -----------------------------
import streamlit as st
from utils.analytics_utils import load_dataframe

st.set_page_config(
    page_title="Swiggy Review Intelligence Platform",
    layout="wide"
)

# -------------------------
# SIDEBAR TITLE FIX
# -------------------------

st.markdown("""
<style>
[data-testid="stSidebarNav"]::before {
    content: "Panel";
    font-size: 30px;
    font-weight: 700;
    margin-left: 10px;
    margin-bottom: 15px;
    display: block;
}
</style>
""", unsafe_allow_html=True)


df = load_dataframe()

total = len(df)
neg_pct = (df["sentiment"]=="Negative").mean()*100
top_aspect = df["aspect"].mode()[0]
avg_conf = df["confidence"].mean()


# =========================
# GLOBAL CSS
# =========================

st.markdown("""
<style>

/* HERO */

.hero-container {
    height: 260px;
}

.hero {
    padding: 50px;
    border-radius: 24px;
    background: linear-gradient(135deg,#ff6b00,#ff9f43);
    color: white;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
}

.hero-sub {
    font-size: 18px;
    opacity: 0.95;
    margin-top: 10px;
}

.logo-box {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(135deg,#ff6b00,#ff9f43);
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* KPI CARDS */

.metric-card {
    padding: 24px;
    border-radius: 18px;
    color: white;
    height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-title {
    font-size: 15px;
    opacity: 0.9;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 34px;
    font-weight: 800;
    line-height: 1.1;
}

.m1 {background: linear-gradient(135deg,#4facfe,#00f2fe);}
.m2 {background: linear-gradient(135deg,#43e97b,#38f9d7);}
.m3 {background: linear-gradient(135deg,#fa709a,#fee140);}
.m4 {background: linear-gradient(135deg,#667eea,#764ba2);}

/* FEATURES */

.feature {
    padding:22px;
    border-radius:16px;
    background:#0b1220;
    border:1px solid #1f2937;
    min-height:140px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HERO SECTION
# =========================

left, right = st.columns([2,1], gap="large")

with left:
    st.markdown("""
    <div class="hero-container">
        <div class="hero">
            <div class="hero-title">
                Swiggy Review Intelligence Platform
            </div>
            <div class="hero-sub">
                AI-powered customer complaint intelligence, semantic search,
                and sentiment analytics built on vector retrieval and RAG.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="hero-container">
        <div class="logo-box">
            <img src="https://upload.wikimedia.org/wikipedia/commons/1/13/Swiggy_logo.png" width="200">
        </div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")


# =========================
# KPI STRIP (FIXED ALIGNMENT)
# =========================

c1, c2, c3, c4 = st.columns(4, gap="large")

c1.markdown(f"""
<div class="metric-card m1">
    <div class="metric-title">Total Reviews</div>
    <div class="metric-value">{total}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="metric-card m2">
    <div class="metric-title">Negative %</div>
    <div class="metric-value">{neg_pct:.1f}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="metric-card m3">
    <div class="metric-title">Top Issue</div>
    <div class="metric-value">{top_aspect}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="metric-card m4">
    <div class="metric-title">Avg Confidence</div>
    <div class="metric-value">{avg_conf:.2f}</div>
</div>
""", unsafe_allow_html=True)


st.write("")
st.divider()


# =========================
# FEATURE GRID
# =========================

st.subheader("Platform Capabilities")

f1,f2,f3 = st.columns(3, gap="large")

with f1:
    st.markdown("""
    <div class="feature">
    <h3>Semantic Vector Search</h3>
    Search customer complaints using FAISS embeddings
    instead of keyword matching for higher recall.
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature">
    <h3>RAG AI Assistant</h3>
    Ask natural language questions and receive
    context-grounded answers from review evidence.
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature">
    <h3>Executive Dashboards</h3>
    Visual sentiment, aspect, and confidence analytics
    for rapid decision making.
    </div>
    """, unsafe_allow_html=True)


f4,f5,f6 = st.columns(3, gap="large")

with f4:
    st.markdown("""
    <div class="feature">
    <h3>Risk Detection</h3>
    Identify high-risk complaint clusters and
    recurring operational failures.
    </div>
    """, unsafe_allow_html=True)

with f5:
    st.markdown("""
    <div class="feature">
    <h3>Evaluation Metrics</h3>
    Precision, confidence distribution, and
    aspect-level breakdown analysis.
    </div>
    """, unsafe_allow_html=True)

with f6:
    st.markdown("""
    <div class="feature">
    <h3>Automated Reports</h3>
    Generate and send intelligence newsletters
    to stakeholders.
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")
st.caption("AI Review Intelligence System — Vector RAG + Sentiment Analytics")


# import streamlit as st
# import pandas as pd
# import pickle
# import faiss
# from collections import Counter

# from langchain_groq import ChatGroq

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore import InMemoryDocstore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document


# # ============================================================
# # 🔑 CONFIG — HARDCODED KEY (as you requested)
# # ============================================================


# INDEX_PATH = "faiss_store/index.faiss"
# META_PATH = "faiss_store/metadata.pkl"


# # ============================================================
# # GROQ RAG CHAIN
# # ============================================================

# def build_rag_chain():

#     

#     llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         api_key=GROQ_API_KEY,
#         temperature=0.2
#     )

#     prompt = ChatPromptTemplate.from_template("""
# You are an AI assistant analyzing Swiggy customer reviews.

# Use ONLY the provided retrieved reviews.
# Do not invent facts.

# Question:
# {question}

# Reviews:
# {context}

# Return:
# 1) Direct answer
# 2) Common pattern observed
# 3) Dominant aspect
# 4) Sentiment trend
# """)

#     return prompt | llm | StrOutputParser()



# # ============================================================
# # FAISS RETRIEVER
# # ============================================================

# @st.cache_resource
# def load_retriever(k=5):

#     with open(META_PATH, "rb") as f:
#         metadata = pickle.load(f)

#     documents = []
#     index_to_docstore_id = {}

#     for i, item in enumerate(metadata):
#         doc_id = str(i)
#         doc = Document(
#             page_content=item["clean_text"],
#             metadata=item
#         )
#         documents.append((doc_id, doc))
#         index_to_docstore_id[i] = doc_id

#     docstore = InMemoryDocstore(dict(documents))

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     index = faiss.read_index(INDEX_PATH)

#     vectorstore = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=docstore,
#         index_to_docstore_id=index_to_docstore_id,
#     )

#     return vectorstore.as_retriever(search_kwargs={"k": k}), metadata


# # ============================================================
# # METRICS
# # ============================================================

# def precision_at_k(docs):
#     if not docs:
#         return 0.0
#     target = docs[0].metadata["aspect"]
#     hits = sum(1 for d in docs if d.metadata["aspect"] == target)
#     return hits / len(docs)


# def load_dataframe():
#     with open(META_PATH, "rb") as f:
#         return pd.DataFrame(pickle.load(f))


# # ============================================================
# # UI
# # ============================================================

# st.set_page_config(layout="wide")
# st.title("Swiggy Review Intelligence System")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# tabs = st.tabs(["AI Assistant", "Evaluation & Insights"])


# # ============================================================
# # TAB 1 — AI ASSISTANT
# # ============================================================

# with tabs[0]:

#     st.subheader("AI Review Assistant")

#     k = st.slider("Top-K Retrieval", 3, 10, 5)

#     retriever, metadata = load_retriever(k)
#     rag_chain = build_rag_chain()

#     query = st.chat_input("Ask about refunds, delivery, support, delays…")

#     if query:

#         docs = retriever.invoke(query)

#         context = "\n".join(
#             f"- {d.page_content} | Aspect={d.metadata['aspect']} | Sentiment={d.metadata['sentiment']}"
#             for d in docs
#         )

#         # ---------- CALL GPT ----------
#         try:
#             answer = rag_chain.invoke({
#                 "question": query,
#                 "context": context
#             })
#         except Exception as e:
#             answer = f"LLM call failed:\n{e}"

#         st.session_state.chat_history.append(("user", query))
#         st.session_state.chat_history.append(("assistant", answer))

#         st.metric("Precision@K", round(precision_at_k(docs), 2))

#         with st.expander("Retrieved Evidence"):
#             for i, d in enumerate(docs, 1):
#                 st.markdown(f"**Result {i}**")
#                 st.write(d.page_content)
#                 st.caption(
#                     f"{d.metadata['aspect']} | "
#                     f"{d.metadata['sentiment']} | "
#                     f"confidence={d.metadata['confidence']}"
#                 )
#                 st.divider()

#     for role, msg in st.session_state.chat_history:
#         st.chat_message(role).write(msg)


# # ============================================================
# # TAB 2 — ANALYTICS
# # ============================================================

# with tabs[1]:

#     st.subheader("Dataset Insights")

#     df = load_dataframe()

#     c1, c2, c3 = st.columns(3)

#     with c1:
#         st.markdown("Sentiment Distribution")
#         st.bar_chart(df["sentiment"].value_counts())

#     with c2:
#         st.markdown("Aspect Distribution")
#         st.bar_chart(df["aspect"].value_counts())

#     with c3:
#         st.markdown("Source Distribution")
#         st.bar_chart(df["source"].value_counts())

#     st.markdown("Sample Reviews")
#     st.dataframe(df.sample(min(10, len(df))))
