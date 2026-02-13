import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = r"D:\Infosys _Springboard\data\final\orchestrated_output.csv"
STORE_DIR = "faiss_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
META_PATH = os.path.join(STORE_DIR, "metadata.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================
# SAFETY CHECKS
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

os.makedirs(STORE_DIR, exist_ok=True)

# ============================================================
# LOAD & VALIDATE DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

REQUIRED_COLUMNS = [
    "clean_text",
    "aspect",
    "sentiment",
    "confidence",
    "source"
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["clean_text"])
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"] != ""]
df = df.reset_index(drop=True)

print(f"Total rows to embed: {len(df)}")

# ============================================================
# EMBEDDING
# ============================================================

model = SentenceTransformer(EMBEDDING_MODEL)

texts = df["clean_text"].tolist()

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ============================================================
# FAISS INDEX (COSINE SIMILARITY)
# ============================================================

dimension = embeddings.shape[1]

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

# ============================================================
# METADATA STORAGE
# ============================================================

metadata = df[REQUIRED_COLUMNS].to_dict(orient="records")

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

# ============================================================
# DONE
# ============================================================

print("FAISS index built successfully")
print("Total vectors stored:", index.ntotal)
