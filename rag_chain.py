from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# CONFIG — PUT YOUR OPENAI KEY HERE
# ============================================================



def build_rag_chain():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant analyzing customer reviews.

You MUST answer ONLY using the retrieved review context.
Do not invent facts not present in the context.

User question:
{question}

Retrieved reviews:
{context}

Return:
1. Direct answer
2. Common pattern observed
3. Dominant aspect
4. Overall sentiment trend

If the context is insufficient, say:
"Not enough evidence in retrieved reviews."
""")

    chain = prompt | llm | StrOutputParser()
    return chain
