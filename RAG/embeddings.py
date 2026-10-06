from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example chunks
chunks = [
    "Students must maintain a minimum attendance of 75%.",
    "Internal assessment includes tests and assignments.",
    "Students must complete the required courses and credits."
]

# Convert text into embeddings
embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk embedding:")
print(embeddings[0])