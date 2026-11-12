
import numpy as np
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking import extract_text
from src.vector_store import create_embedding


# =====================================
# 1. EVALUATION QUESTIONS
# =====================================

questions = [
    ("What medical benefits do employees receive?", "employee_policy.txt"),
    ("What are normal working hours?", "employee_policy.txt"),
    ("What responsibilities do employees have?", "employee_policy.txt"),

    ("How should employees handle fire emergencies?", "safety_manual.txt"),
    ("Where are first aid kits available?", "safety_manual.txt"),
    ("What should employees do during evacuation?", "safety_manual.txt"),

    ("How should employees protect passwords?", "security_manual.txt"),
    ("Why is multi-factor authentication important?", "security_manual.txt"),
    ("How should company information be protected?", "security_manual.txt"),

    ("How can employees apply for annual leave?", "leave_policy.docx"),
    ("What is the sick leave policy?", "leave_policy.docx"),
    ("Which public holidays does the company observe?", "leave_policy.docx"),

    ("What behavior is expected from employees?", "company_rules.txt"),
    ("What are the company attendance rules?", "company_rules.txt"),
    ("How should company documents be protected?", "company_rules.txt")
]


# =====================================
# 2. LOAD DOCUMENT CHUNKS
# =====================================

def load_chunks(chunk_size):

    all_chunks = []

    for file_path in Path("documents").iterdir():

        if file_path.suffix.lower() not in [".txt", ".docx", ".pdf"]:
            continue

        pages = extract_text(str(file_path))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(50, chunk_size - 1),
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        for page in pages:

            texts = splitter.split_text(page["text"])

            for text in texts:

                all_chunks.append({
                    "source": file_path.name,
                    "page": page["page"],
                    "text": text
                })

    return all_chunks


# =====================================
# 3. COSINE SIMILARITY
# =====================================

def cosine_scores(query_vector, vectors):

    query_vector = np.asarray(query_vector, dtype=np.float32)

    vectors = np.asarray(vectors, dtype=np.float32)

    denominator = (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    return np.divide(
        vectors @ query_vector,
        denominator,
        out=np.zeros(len(vectors), dtype=np.float32),
        where=denominator != 0
    )


# =====================================
# 4. EVALUATION
# =====================================

def evaluate(chunks, vectors, query_vectors):

    hit1 = 0
    hit3 = 0
    hit5 = 0

    reciprocal_ranks = []

    for i, (question, expected_source) in enumerate(questions):

        scores = cosine_scores(
            query_vectors[i],
            vectors
        )

        ranked_indices = np.argsort(scores)[::-1]

        first_rank = None

        for rank, index in enumerate(ranked_indices, start=1):

            chunk = chunks[index]

            if chunk["source"] == expected_source:

                first_rank = rank
                break

        if first_rank is not None:

            if first_rank <= 1:
                hit1 += 1

            if first_rank <= 3:
                hit3 += 1

            if first_rank <= 5:
                hit5 += 1

            reciprocal_ranks.append(1 / first_rank)

        else:
            reciprocal_ranks.append(0)

    total = len(questions)

    return {
        "Hit@1": hit1 / total,
        "Hit@3": hit3 / total,
        "Hit@5": hit5 / total,
        "MRR": sum(reciprocal_ranks) / total
    }


# =====================================
# 5. RUN CONFIGURATION
# =====================================

def run_configuration(size, query_vectors):

    print(f"\nTesting chunk size: {size}")

    chunks = load_chunks(size)

    if not chunks:
        raise ValueError("No document chunks found")

    vectors = np.asarray(
        [create_embedding(c["text"]) for c in chunks],
        dtype=np.float32
    )

    return evaluate(chunks, vectors, query_vectors)


# =====================================
# 6. MAIN
# =====================================

if __name__ == "__main__":

    print("Starting retrieval evaluation...")

    query_vectors = np.asarray(
        [create_embedding(q) for q, _ in questions],
        dtype=np.float32
    )

    config_a = run_configuration(256, query_vectors)

    config_b = run_configuration(512, query_vectors)

    print("\n========== RESULTS ==========")

    print(f"{'Metric':<15}{'Config A':<15}{'Config B':<15}")

    for metric in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:

        print(
            f"{metric:<15}"
            f"{config_a[metric]:<15.3f}"
            f"{config_b[metric]:<15.3f}"
        )

    print("\nBest configuration:")

    if config_a["MRR"] > config_b["MRR"]:
        print("Config A - Chunk size 256")

    elif config_b["MRR"] > config_a["MRR"]:
        print("Config B - Chunk size 512")

    else:
        print("Both configurations have equal MRR")

    print("\nRETRIEVAL EVALUATION COMPLETED")
