import json
from sentence_transformers import SentenceTransformer

# Input and output files
INPUT_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/processed/embedded_chunks.json"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")

# Extract text
texts = [chunk["text"] for chunk in chunks]

# Create embeddings
print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Add embedding to every chunk
for chunk, embedding in zip(chunks, embeddings):
    chunk["embedding"] = embedding.tolist()

# Save embedded chunks
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nEmbedding process complete.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")