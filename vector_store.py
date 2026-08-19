import json
import chromadb

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/embedded_chunks.json"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "linux_documents"


# ============================================================
# LOAD EMBEDDED CHUNKS
# ============================================================

print("Loading embedded chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# CREATE CHROMA DATABASE
# ============================================================

print("Creating ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "Linux documentation for RAG assistant"
    }
)


# ============================================================
# PREPARE DATA
# ============================================================

ids = []
documents = []
embeddings = []
metadatas = []

for chunk in chunks:

    ids.append(chunk["chunk_id"])

    documents.append(chunk["text"])

    embeddings.append(chunk["embedding"])

    metadatas.append({
        "source": chunk["source"],
        "page": chunk["page"]
    })


# ============================================================
# INSERT INTO CHROMADB
# ============================================================

print("Adding documents to ChromaDB...")

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


# ============================================================
# VERIFY
# ============================================================

count = collection.count()

print("\nVector database created successfully.")
print(f"Documents stored: {count}")
print(f"Database location: {CHROMA_PATH}")
print(f"Collection: {COLLECTION_NAME}")