import streamlit as st
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
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Linux RAG Assistant",
    page_icon="🐧",
    layout="wide"
)


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# Connect to ChromaDB
# ---------------------------------------------------------

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ---------------------------------------------------------
# Retrieve documents
# ---------------------------------------------------------

def retrieve_documents(question, embedding_model, collection):

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
# Build prompt
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

    return prompt


# ---------------------------------------------------------
# Ask Ollama
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
# Load RAG system
# ---------------------------------------------------------

with st.spinner("Loading Linux RAG system..."):

    embedding_model = load_embedding_model()
    collection = load_collection()


# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------

st.title("🐧 Linux RAG Assistant")

st.write(
    "Ask questions about Linux commands, permissions, "
    "users, processes, networking, Bash, and system administration."
)

st.info(
    "Answers are generated using your Linux documentation "
    "and a local Ollama LLM."
)


question = st.text_input(
    "Ask a Linux question:",
    placeholder="Example: How do I change file permissions in Linux?"
)


if st.button("Ask Linux Assistant"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving documents and generating answer..."):

            try:

                retrieved_documents = retrieve_documents(
                    question,
                    embedding_model,
                    collection
                )

                prompt = build_prompt(
                    question,
                    retrieved_documents
                )

                answer = ask_ollama(prompt)

                st.subheader("Answer")

                st.write(answer)

                st.subheader("Retrieved Sources")

                for index, doc in enumerate(
                    retrieved_documents,
                    1
                ):

                    with st.expander(
                        f"Source {index}: "
                        f"{doc['source']} | "
                        f"Page: {doc['page']} | "
                        f"Distance: {doc['distance']:.4f}"
                    ):

                        st.write(doc["text"])


            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running."
                )

            except Exception as error:

                st.error(
                    f"Something went wrong: {error}"
                )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Project Information")

    st.write("**Project:** Linux RAG Assistant")

    st.write("**Documents:** 5 PDFs")

    st.write("**Chunks:** 46")

    st.write(
        "**Embedding:** "
        "all-MiniLM-L6-v2"
    )

    st.write(
        "**Vector Database:** ChromaDB"
    )

    st.write(
        "**Local LLM:** llama3.2:1b"
    )

    st.write(
        "**LLM Runner:** Ollama"
    )