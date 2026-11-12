
import os
import re
import json
import hashlib

from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

import fitz
from docx import Document
import tiktoken

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ==========================================
# 1. DOCUMENT TEXT EXTRACTION
# ==========================================

def extract_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    pages = []

    if extension == ".pdf":

        pdf = fitz.open(file_path)

        for page_number, page in enumerate(pdf, start=1):

            pages.append({
                "page": page_number,
                "text": page.get_text()
            })

        pdf.close()

    elif extension == ".docx":

        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        pages.append({
            "page": 1,
            "text": text
        })

    elif extension == ".txt":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        pages.append({
            "page": 1,
            "text": text
        })

    else:

        raise ValueError(
            "Only PDF, DOCX and TXT supported"
        )

    return pages


# ==========================================
# 2. FIXED-SIZE TOKEN CHUNKING
# ==========================================

encoding = tiktoken.get_encoding("cl100k_base")


def fixed_token_chunk(
    text,
    chunk_size=512,
    overlap=50
):

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk_text = encoding.decode(chunk_tokens)

        if chunk_text.strip():

            chunks.append(chunk_text)

        if end >= len(tokens):

            break

        start = end - overlap

    return chunks


# ==========================================
# 3. SENTENCE BASED CHUNKING
# ==========================================

def sentence_chunk(text, max_words=100):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    chunks = []

    current_chunk = []
    current_words = 0

    for sentence in sentences:

        words = len(sentence.split())

        if current_words + words > max_words and current_chunk:

            chunks.append(" ".join(current_chunk))

            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)

        current_words += words

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


# ==========================================
# 4. RECURSIVE CHARACTER TEXT SPLITTER
# ==========================================

def recursive_chunk(
    text,
    chunk_size=512,
    overlap=50
):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    )

    return splitter.split_text(text)


# ==========================================
# 5. STABLE CHUNK ID
# ==========================================

def generate_chunk_id(
    source,
    page,
    text
):

    # Content-based hash.
    # Same source, page and content = same ID.

    unique_string = (
        f"{source}|{page}|{text.strip()}"
    )

    return str(
        uuid5(
            NAMESPACE_URL,
            unique_string
        )
    )


# ==========================================
# 6. METADATA CREATION
# ==========================================

def create_metadata(
    source,
    page,
    section,
    chunk_index,
    text
):

    extension = os.path.splitext(source)[1]

    metadata = {

        "source": source,

        "page": page,

        "section": section,

        "created_date": datetime.now().isoformat(),

        "chunk_index": chunk_index,

        "doc_type": extension.replace(".", "").upper(),

        "chunk_id": generate_chunk_id(
            source,
            page,
            text
        )

    }

    return metadata


# ==========================================
# 7. SECTION HEADING DETECTION
# ==========================================

def detect_section(text):

    for line in text.splitlines():

        line = line.strip()

        if (
            line
            and len(line) < 100
            and not line.endswith(".")
        ):

            return line

    return "General"


# ==========================================
# 8. CITATION STRING
# ==========================================

def describe_chunk(chunk):

    metadata = chunk["metadata"]

    return (
        f"Source: {metadata['source']} | "
        f"Page: {metadata['page']} | "
        f"Section: {metadata['section']} | "
        f"Chunk: {metadata['chunk_index']}"
    )


# ==========================================
# 9. PROCESS ONE DOCUMENT
# ==========================================

def process_document(file_path):

    source = os.path.basename(file_path)

    pages = extract_text(file_path)

    all_chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = page_data["text"]

        chunks = recursive_chunk(
            text,
            chunk_size=512,
            overlap=50
        )

        for index, chunk_text in enumerate(chunks):

            section = detect_section(chunk_text)

            metadata = create_metadata(
                source,
                page_number,
                section,
                index,
                chunk_text
            )

            all_chunks.append({

                "id": metadata["chunk_id"],

                "text": chunk_text,

                "metadata": metadata

            })

    return all_chunks


# ==========================================
# 10. PROCESS ALL DOCUMENTS
# ==========================================

def process_all_documents(folder="documents"):

    all_chunks = []

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        if not filename.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):

            continue

        print("\nProcessing:", filename)

        chunks = process_document(file_path)

        print("Total chunks:", len(chunks))

        all_chunks.extend(chunks)

    return all_chunks


# ==========================================
# 11. CHUNKING COMPARISON
# ==========================================

def compare_chunking(text):

    fixed = fixed_token_chunk(
        text,
        512,
        50
    )

    sentence = sentence_chunk(text)

    recursive = recursive_chunk(
        text,
        512,
        50
    )

    print("\nCHUNKING COMPARISON")

    print("Fixed token chunks:", len(fixed))

    print("Sentence chunks:", len(sentence))

    print("Recursive chunks:", len(recursive))


# ==========================================
# 12. STABLE ID TEST
# ==========================================

def stable_id_test(file_path):

    first = process_document(file_path)

    second = process_document(file_path)

    first_ids = {
        chunk["id"] for chunk in first
    }

    second_ids = {
        chunk["id"] for chunk in second
    }

    print("\nSTABLE ID TEST")

    print("First run:", len(first_ids))

    print("Second run:", len(second_ids))

    print(
        "Same IDs:",
        first_ids == second_ids
    )


# ==========================================
# 13. INCREMENTAL CHANGE TEST
# ==========================================

def incremental_test():

    source = "company_rules.txt"

    original = "Company employees must follow security policies."

    changed = "Company employees must follow updated security policies."

    original_id = generate_chunk_id(
        source,
        1,
        original
    )

    changed_id = generate_chunk_id(
        source,
        1,
        changed
    )

    print("\nINCREMENTAL INDEXING TEST")

    print("Original ID:", original_id)

    print("Changed ID:", changed_id)

    print("ID changed:", original_id != changed_id)


# ==========================================
# 14. MAIN
# ==========================================

if __name__ == "__main__":

    folder = "documents"

    all_chunks = process_all_documents(folder)

    print("\nTOTAL DOCUMENT CHUNKS:", len(all_chunks))

    if all_chunks:

        print("\nFIRST CHUNK")

        print("Text:", all_chunks[0]["text"][:300])

        print(
            "Citation:",
            describe_chunk(all_chunks[0])
        )

        print(
            "Metadata:",
            json.dumps(
                all_chunks[0]["metadata"],
                indent=4
            )
        )

        sample_text = all_chunks[0]["text"]

        compare_chunking(sample_text)

        first_file = os.path.join(
            folder,
            os.listdir(folder)[0]
        )

        stable_id_test(first_file)

    incremental_test()

    print("\nTASK 2 COMPLETED")
