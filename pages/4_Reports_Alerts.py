import streamlit as st
from utils.analytics_utils import load_dataframe
from utils.email_utils import send_newsletter
from utils.rag_utils import build_rag_chain

st.title("Reports & Alerts")

df = load_dataframe()
rag_chain = build_rag_chain()

aspect = st.selectbox("Select Aspect", df["aspect"].unique())

subset = df[df.aspect == aspect]

context = "\n".join(subset.clean_text.head(20).tolist())

if st.button("Generate Newsletter Summary"):

    summary = rag_chain.invoke({
        "question": f"Summarize customer issues about {aspect}",
        "context": context
    })

    st.session_state["newsletter"] = summary

if "newsletter" in st.session_state:
    st.markdown("### Newsletter Preview")
    st.write(st.session_state["newsletter"])

emails = st.text_area(
    "Recipient Emails (comma separated)"
)

if st.button("Send Newsletter"):

    to_list = [e.strip() for e in emails.split(",") if e.strip()]

    send_newsletter(
        to_list,
        subject=f"Customer Insight Report — {aspect}",
        body=st.session_state["newsletter"]
    )

    st.success("Emails sent successfully")
