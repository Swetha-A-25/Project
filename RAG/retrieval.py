from sentence_transformers import SentenceTransformer
import faiss
import pickle


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS vector database
index = faiss.read_index(
    "vectorstore/college_documents.index"
)


# Load the original chunks
with open(
    "vectorstore/chunks.pkl",
    "rb"
) as file:
    chunks = pickle.load(file)


def search_documents(question, top_k=3):

    # Convert the student's question into an embedding
    question_embedding = model.encode(
        [question]
    )

    # Search FAISS
    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        results.append({
            "text": chunks[index_number]["text"],
            "page": chunks[index_number]["page"],
            "distance": float(distance)
        })

    return results


# Test question
question = "What is the minimum attendance requirement?"

results = search_documents(question)


print("\nQUESTION:")
print(question)

print("\nRELEVANT RESULTS:")

for i, result in enumerate(results, start=1):

    print("\n==============================")
    print("RESULT", i)
    print("PAGE:", result["page"])
    print("DISTANCE:", result["distance"])
    print("==============================")
    print(result["text"])