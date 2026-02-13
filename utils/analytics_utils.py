import pickle
import pandas as pd
from collections import Counter

META_PATH = r"D:\Infosys _Springboard\vector_db\faiss_store\metadata.pkl"


def load_dataframe():
    with open(META_PATH, "rb") as f:
        return pd.DataFrame(pickle.load(f))


def precision_at_k(docs):
    if not docs:
        return 0.0
    target = docs[0].metadata["aspect"]
    hits = sum(1 for d in docs if d.metadata["aspect"] == target)
    return hits / len(docs)


def sentiment_counts(df):
    return df["sentiment"].value_counts()


def aspect_counts(df):
    return df["aspect"].value_counts()


def source_counts(df):
    return df["source"].value_counts()
