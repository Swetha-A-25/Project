from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle


# -----------------------------
# 1. Read PDF
# -----------------------------

pdf_path = "documents/Academic_Regulations_Demo.pdf"

reader = PdfReader(pdf_path)

chunks = []

chunk_size = 500
overlap = 100

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    text = text.strip()

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "page": page_number
        })

        start = end - overlap


print("Total chunks:", len(chunks))


# -----------------------------
# 2. Create embeddings
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)


# -----------------------------
# 3. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# -----------------------------
# 4. Save FAISS database
# -----------------------------

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(
    index,
    "vectorstore/college_documents.index"
)


# -----------------------------
# 5. Save chunk information
# -----------------------------

with open(
    "vectorstore/chunks.pkl",
    "wb"
) as file:

    pickle.dump(chunks, file)


print("FAISS vector database saved successfully!")