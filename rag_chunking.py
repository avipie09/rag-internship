import os
import re
import json
from pypdf import PdfReader

RAW_DOCUMENTS_DIR = "data/raw_documents"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks.json")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    return text

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            start = 0

            while start < len(paragraph):
                end = start + chunk_size
                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                start = end - overlap

            continue

        proposed = (
            current_chunk + "\n\n" + paragraph
            if current_chunk
            else paragraph
        )

        if len(proposed) <= chunk_size:
            current_chunk = proposed
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_pdf(pdf_path: str) -> list[dict]:
    print(f"\nProcessing: {pdf_path}")

    reader = PdfReader(pdf_path)
    document_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        cleaned = clean_text(text)

        if not cleaned:
            continue

        chunks = split_text(
            cleaned,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk in chunks:
            document_chunks.append({
                "text": chunk,
                "source": os.path.basename(pdf_path),
                "page": page_number
            })

    return document_chunks

def main():
    all_chunks = []

    if not os.path.exists(RAW_DOCUMENTS_DIR):
        print(f"ERROR: Folder not found: {RAW_DOCUMENTS_DIR}")
        return

    pdf_files = [
        file
        for file in os.listdir(RAW_DOCUMENTS_DIR)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("ERROR: No PDF files found.")
        print(f"Put your PDFs inside: {RAW_DOCUMENTS_DIR}")
        return

    print("=" * 60)
    print("        LINUX RAG DOCUMENT PROCESSING")
    print("=" * 60)

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(
            RAW_DOCUMENTS_DIR,
            pdf_file
        )

        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

        print(f"Created {len(chunks)} chunks.")

    final_chunks = []

    for index, chunk in enumerate(all_chunks, start=1):
        final_chunks.append({
            "chunk_id": f"chunk_{index:05d}",
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"]
        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            final_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nTotal chunks: {len(final_chunks)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFirst 3 chunks:\n")

    for chunk in final_chunks[:3]:
        print("-" * 60)
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print("\nText:")
        print(chunk["text"][:500])

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
    