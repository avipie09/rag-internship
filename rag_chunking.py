import os
import re
from pypdf import PdfReader

# Using the specific file you uploaded in your project folder
PDF_FILENAME = "Permissions_User_Management.pdf"

if not os.path.exists(PDF_FILENAME):
    print(f"\n Error: '{PDF_FILENAME}' not found in this folder!")
    print("Please make sure you copied the PDF file into: /home/avi/projects/rag-internship/")
    exit()

print(f" Extracting text from {PDF_FILENAME}...")
reader = PdfReader(PDF_FILENAME)
raw_text = ""
for page in reader.pages:
    text_content = page.extract_text()
    if text_content:
        raw_text += text_content + "\n"

#  Cleaning Pipeline (Assignment Part 2)
def clean_text(text: str) -> str:
    # Remove PDF structural layout artifacts (headers, footers, page splits)
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"--- PAGE \d+ ---", "", text)
    text = re.sub(r"Copyright © \d+.*", "", text)
    
    text = text.replace("\r\n", "\n")
    lines = [line.strip() for line in text.split("\n")]
    valid_lines = [line for line in lines if line]
    return "\n\n".join(valid_lines)

cleaned_text = clean_text(raw_text)

#  Recursive Paragraph Chunker (Assignment Part 3)
def recursive_chunker(text: str, chunk_size: int = 500) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_block = ""
    
    for paragraph in paragraphs:
        if len(current_block) + len(paragraph) <= chunk_size:
            current_block = f"{current_block}\n\n{paragraph}".strip()
        else:
            if current_block:
                chunks.append(current_block)
            current_block = paragraph
            
    if current_block:
        chunks.append(current_block)
    return chunks

processed_chunks = recursive_chunker(cleaned_text, chunk_size=400)

print(f" Document successfully parsed into {len(processed_chunks)} chunks.")
print("\n========================================================")
print("             YOUR 3 WEEK 3 REPORT CHUNKS")
print("========================================================\n")

for i, chunk in enumerate(processed_chunks[:3], 1):
    print(f"--- CHUNK {i} ---")
    print(chunk)
    print("\n" + "="*50 + "\n")
