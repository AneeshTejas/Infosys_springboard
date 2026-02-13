import streamlit as st
from utils.retriever_utils import load_retriever
from utils.rag_utils import build_rag_chain
from utils.analytics_utils import precision_at_k

st.title("AI Review Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

k = st.slider("Top-K Retrieval", 3, 10, 5)

retriever, _ = load_retriever(k)
rag_chain = build_rag_chain()

query = st.chat_input("Ask about refunds, delivery, support issues…")

if query:

    docs = retriever.invoke(query)

    context = "\n".join(
        f"- {d.page_content} | {d.metadata['aspect']} | {d.metadata['sentiment']}"
        for d in docs
    )

    answer = rag_chain.invoke({
        "question": query,
        "context": context
    })

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", answer))

    st.metric("Precision@K", round(precision_at_k(docs), 2))

    with st.expander("Retrieved Evidence"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Result {i}**")
            st.write(d.page_content)
            st.caption(d.metadata)
            st.divider()

for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)
