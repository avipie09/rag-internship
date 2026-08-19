import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "linux_documents"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD MODELS AND DATABASE
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Database contains {collection.count()} documents.")


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query: str, top_k: int = TOP_K):

    # Convert user question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# ============================================================
# TEST RETRIEVAL
# ============================================================

if __name__ == "__main__":

    question = input("\nAsk a Linux question: ")

    results = retrieve_documents(question)

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\n--- RESULT {i} ---")

        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Distance: {distance:.4f}")

        print("\nText:")
        print(document)

        print("\n" + "-" * 70)