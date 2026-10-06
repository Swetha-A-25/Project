from pypdf import PdfReader

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

for i, chunk in enumerate(chunks, start=1):

    print("\n==============================")
    print("CHUNK", i)
    print("PAGE:", chunk["page"])
    print("==============================")

    print(chunk["text"])