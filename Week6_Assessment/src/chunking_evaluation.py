
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking import extract_text

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = {
    "employee_policy.txt": [
        "medical insurance employee health benefits",
        "working hours attendance rules"
    ],
    "safety_manual.txt": [
        "fire safety emergency exits",
        "first aid workplace injuries"
    ],
    "security_manual.txt": [
        "password security strong passwords",
        "data protection confidential information"
    ],
    "leave_policy.docx": [
        "annual leave manager approval",
        "sick leave medical documentation"
    ],
    "company_rules.txt": [
        "employee code of conduct",
        "company confidentiality rules"
    ]
}

questions = [
    ("What medical benefits do employees receive?", "employee_policy.txt"),
    ("What are normal working hours?", "employee_policy.txt"),
    ("How should employees handle fire emergencies?", "safety_manual.txt"),
    ("Where are first aid kits available?", "safety_manual.txt"),
    ("How should employees protect passwords?", "security_manual.txt"),
    ("What are company data protection rules?", "security_manual.txt"),
    ("How can employees apply for annual leave?", "leave_policy.docx"),
    ("What is the sick leave policy?", "leave_policy.docx"),
    ("What behavior is expected from employees?", "company_rules.txt"),
    ("How should company information be protected?", "company_rules.txt")
]

chunk_sizes = [128, 256, 512, 1024]


def get_chunks(size):

    all_chunks = []

    for filename in documents:

        pages = extract_text("documents/" + filename)

        for page in pages:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            chunks = splitter.split_text(page["text"])

            for chunk in chunks:

                all_chunks.append({
                    "source": filename,
                    "page": page["page"],
                    "text": chunk
                })

    return all_chunks


def evaluate(size):

    chunks = get_chunks(size)

    chunk_vectors = model.encode(
        [item["text"] for item in chunks]
    )

    correct = 0

    for question, expected_source in questions:

        query_vector = model.encode([question])

        scores = cosine_similarity(
            query_vector,
            chunk_vectors
        )[0]

        best_index = np.argmax(scores)

        result = chunks[best_index]

        if result["source"] == expected_source:
            correct += 1

    hit_rate = correct / len(questions)

    return len(chunks), correct, hit_rate


if __name__ == "__main__":

    print("\nCHUNK SIZE EXPERIMENT")

    print("-" * 60)

    print(
        f"{'Chunk Size':<15}"
        f"{'Chunks':<15}"
        f"{'Correct':<15}"
        f"{'Hit Rate':<15}"
    )

    for size in chunk_sizes:

        total, correct, hit_rate = evaluate(size)

        print(
            f"{size:<15}"
            f"{total:<15}"
            f"{correct}/10{'':<10}"
            f"{hit_rate:.2f}"
        )

    print("-" * 60)
