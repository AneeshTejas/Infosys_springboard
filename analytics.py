import pickle
import pandas as pd
from collections import Counter

META_PATH = "faiss_store/metadata.pkl"

def load_dataframe():
    with open(META_PATH, "rb") as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

def sentiment_distribution(df):
    return Counter(df["sentiment"])

def aspect_distribution(df):
    return Counter(df["aspect"])

def source_distribution(df):
    return Counter(df["source"])

