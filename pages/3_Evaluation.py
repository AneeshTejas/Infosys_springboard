import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from utils.analytics_utils import load_dataframe
from utils.rag_utils import load_faiss_retriever_only


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(layout="wide")
st.title("Evaluation & Intelligence Dashboard")

df = load_dataframe()


# =========================
# COLOR KPI CARDS CSS
# =========================

st.markdown("""
<style>
.kpi {
 padding:18px;
 border-radius:18px;
 color:white;
 font-weight:700;
}
.kpi1 {background: linear-gradient(135deg,#667eea,#764ba2);}
.kpi2 {background: linear-gradient(135deg,#43e97b,#38f9d7);}
.kpi3 {background: linear-gradient(135deg,#fa709a,#fee140);}
.kpi4 {background: linear-gradient(135deg,#4facfe,#00f2fe);}
.block {
 padding:16px;
 border-radius:14px;
 background:#0f172a;
}
</style>
""", unsafe_allow_html=True)


# =========================
# KPI METRICS
# =========================

pos_pct = (df["sentiment"]=="Positive").mean()*100
neu_pct = (df["sentiment"]=="Neutral").mean()*100
neg_pct = (df["sentiment"]=="Negative").mean()*100
avg_conf = df["confidence"].mean()

c1,c2,c3,c4 = st.columns(4)

c1.markdown(f'<div class="kpi kpi1">Positive %<h2>{pos_pct:.1f}</h2></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi kpi2">Neutral %<h2>{neu_pct:.1f}</h2></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi kpi3">Negative %<h2>{neg_pct:.1f}</h2></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="kpi kpi4">Avg Confidence<h2>{avg_conf:.2f}</h2></div>', unsafe_allow_html=True)

st.divider()


# =========================
# DONUT + CONF HISTOGRAM
# =========================

colA, colB = st.columns(2)

with colA:
    sent_counts = df["sentiment"].value_counts().reset_index()
    sent_counts.columns = ["sentiment","count"]

    fig = px.pie(
        sent_counts,
        names="sentiment",
        values="count",
        hole=0.6,
        color="sentiment",
        color_discrete_map={
            "Positive":"#22c55e",
            "Neutral":"#eab308",
            "Negative":"#ef4444"
        },
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


with colB:
    fig = px.histogram(
        df,
        x="confidence",
        nbins=25,
        title="Confidence Distribution",
        color_discrete_sequence=["#60a5fa"]
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ASPECT vs SENTIMENT STACKED
# =========================

st.subheader("Aspect vs Sentiment Breakdown")

pivot = pd.crosstab(df["aspect"], df["sentiment"])

fig = go.Figure()

colors = {
    "Positive":"#22c55e",
    "Neutral":"#eab308",
    "Negative":"#ef4444"
}

for col in pivot.columns:
    fig.add_bar(
        name=col,
        x=pivot.index,
        y=pivot[col],
        marker_color=colors[col]
    )

fig.update_layout(barmode="stack", height=450)
st.plotly_chart(fig, use_container_width=True)


# =========================
# LOW CONFIDENCE TABLE
# =========================

st.subheader("Low Confidence Reviews")

low_df = df.sort_values("confidence").head(15)
st.dataframe(low_df)


# =========================
# VECTOR SEARCH (NO LLM)
# =========================

st.divider()
st.subheader("Query Explorer")

retriever = load_faiss_retriever_only(k=5)

query = st.text_input("Search similar reviews")

if query:
    docs = retriever.invoke(query)

    for i,d in enumerate(docs,1):
        st.markdown(f"**Result {i}**")
        st.write(d.page_content)
        st.caption(
            f"Aspect: {d.metadata['aspect']} | "
            f"Sentiment: {d.metadata['sentiment']} | "
            f"Confidence: {d.metadata['confidence']}"
        )
        st.divider()
