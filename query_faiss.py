import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# ============================================================
# LOAD INDEX & METADATA
# ============================================================

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

print("Index loaded.")
print("Total vectors in index:", index.ntotal)

model = SentenceTransformer(EMBEDDING_MODEL)

# ============================================================
# QUERY LOOP
# ============================================================

while True:
    query = input("\nEnter query (or 'exit'): ").strip()
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, TOP_K)

    print("\nTop results:\n")

    for rank, idx in enumerate(indices[0], start=1):
        item = metadata[idx]
        print(f"Rank {rank}")
        print("Text:", item["clean_text"])
        print(
            "Meta:",
            {
                "aspect": item["aspect"],
                "sentiment": item["sentiment"],
                "confidence": item["confidence"],
                "source": item["source"],
            }
        )
        print("-" * 60)
