from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test sentences
sentences = [
    "The delivery was late and customer support was unhelpful",
    "Refund has not been processed even after cancellation",
    "The app crashes during payment"
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("Sample vector (first 5 values):", embeddings[0][:5])
