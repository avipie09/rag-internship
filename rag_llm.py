import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "linux_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/chat"

TOP_K = 3


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# Connect to ChromaDB
# ---------------------------------------------------------

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Database contains {collection.count()} documents.")


# ---------------------------------------------------------
# Retrieve relevant documents
# ---------------------------------------------------------

def retrieve_documents(question):

    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved.append({
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": distance
        })

    return retrieved


# ---------------------------------------------------------
# Build RAG prompt
# ---------------------------------------------------------

def build_prompt(question, retrieved_documents):

    context_parts = []

    for index, doc in enumerate(retrieved_documents, 1):
        context_parts.append(
            f"""
SOURCE {index}
File: {doc['source']}
Page: {doc['page']}

Content:
{doc['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful Linux documentation assistant.

Answer the user's question using the documentation below.

DOCUMENTATION:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Use the documentation to answer the question.
- Do not make up information.
- If the documentation contains the answer, give the answer directly.
- If a command appears in the documentation, show the command.
- Keep the answer short and clear.
- Mention the relevant source file and page.

ANSWER:
"""

    return prompt# ---------------------------------------------------------
# Send prompt to Ollama
# ---------------------------------------------------------

def ask_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# ---------------------------------------------------------
# Main RAG loop
# ---------------------------------------------------------

print()
print("=" * 70)
print("             LINUX RAG ASSISTANT")
print("=" * 70)

while True:

    question = input(
        "\nAsk a Linux question (or type 'exit'): "
    ).strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    print("\nRetrieving relevant documents...")

    retrieved_documents = retrieve_documents(question)

    print("\nGenerating answer using local LLM...")

    prompt = build_prompt(
        question,
        retrieved_documents
    )

    try:

        answer = ask_ollama(prompt)

        print("\n" + "=" * 70)
        print("                         ANSWER")
        print("=" * 70)

        print(answer)

        print("\n" + "=" * 70)
        print("                    RETRIEVED SOURCES")
        print("=" * 70)

        for index, doc in enumerate(
            retrieved_documents,
            1
        ):

            print(
                f"\n{index}. {doc['source']} "
                f"| Page: {doc['page']} "
                f"| Distance: {doc['distance']:.4f}"
            )

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Could not connect to Ollama."
        )

        print(
            "Make sure Ollama is running."
        )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )