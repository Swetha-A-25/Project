from retrieval import search_documents
import ollama


def answer_question(question):

    # 1. Retrieve relevant information
    results = search_documents(question, top_k=3)

    if not results:
        return "Sorry, I could not find relevant information in the documents."

    # 2. Combine retrieved document content
    context = ""

    for result in results:
        context += (
            f"Page {result['page']}:\n"
            f"{result['text']}\n\n"
        )

    # 3. Give the retrieved information to Gemma
    prompt = f"""
You are SecureRAG, an institutional document assistant.

Answer the student's question ONLY using the information
provided in the document context below.

Do not use outside knowledge.
Do not guess or invent information.

If the answer is not available in the context, say:
"I could not find this information in the verified documents."

Student question:
{question}

Document context:
{context}

Give a short, clear answer.
"""

    # 4. Ask Gemma to generate the answer
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # 5. Add source
    source_pages = sorted(
        set(result["page"] for result in results)
    )

    return (
        f"Answer:\n{answer}\n\n"
        f"Source: Page(s) {', '.join(map(str, source_pages))}"
    )


# -----------------------------
# Test SecureRAG
# -----------------------------

question = "What is the minimum attendance requirement?"

response = answer_question(question)

print("\n==============================")
print("SECURERAG ANSWER")
print("==============================")
print(response)