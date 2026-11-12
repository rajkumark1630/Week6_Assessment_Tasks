
import os
import time
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from openai import OpenAI


# ==========================================
# 1. INITIAL SETUP
# ==========================================

load_dotenv()

print("=" * 60)
print("WEEK 6 - TASK 1: EMBEDDINGS AND COSINE SIMILARITY")
print("=" * 60)

sentences = [
    "The cat is sleeping on the sofa.",
    "A kitten is resting on a couch.",
    "Python is a popular programming language.",
    "Java is used for software development.",
    "The weather is sunny today.",
    "It is a bright and warm day.",
    "Machine learning uses data to make predictions.",
    "AI systems can learn from examples.",
    "The company provides employee health insurance.",
    "Staff members receive medical benefits."
]


# ==========================================
# 2. LOCAL EMBEDDING
# ==========================================

print("\n[1] LOCAL EMBEDDING")

model = SentenceTransformer("all-MiniLM-L6-v2")

start_time = time.time()

local_vectors = model.encode(
    sentences,
    convert_to_numpy=True
)

local_time = time.time() - start_time

print("Model: all-MiniLM-L6-v2")
print("Dimensions:", local_vectors.shape)
print("Embedding time:", round(local_time, 4), "seconds")


# ==========================================
# 3. OPENROUTER EMBEDDING API
# ==========================================

print("\n[2] OPENROUTER EMBEDDING")

api_vectors = None
api_time = None

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key:

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        start_time = time.time()

        response = client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=sentences
        )

        api_vectors = np.array([
            item.embedding
            for item in sorted(
                response.data,
                key=lambda x: x.index
            )
        ])

        api_time = time.time() - start_time

        print("Model: openai/text-embedding-3-small")
        print("Dimensions:", api_vectors.shape)
        print("Embedding time:", round(api_time, 4), "seconds")

    except Exception as error:
        print("OpenRouter API error:", error)

else:
    print("OPENROUTER_API_KEY not found in .env")


# ==========================================
# 4. MANUAL COSINE SIMILARITY
# ==========================================

print("\n[3] MANUAL COSINE SIMILARITY")


def cosine_similarity(a, b):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Vectors must have same dimensions")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


test_cases = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0])
]

for a, b in test_cases:

    score = cosine_similarity(a, b)

    print(
        a, "vs", b,
        "->", round(score, 2)
    )


# ==========================================
# 5. SCIKIT-LEARN COMPARISON
# ==========================================

print("\n[4] SCIKIT-LEARN COMPARISON")

for a, b in test_cases:

    manual_score = cosine_similarity(a, b)

    sklearn_score = sklearn_cosine(
        [a], [b]
    )[0][0]

    print("Manual:", round(manual_score, 2))
    print("Sklearn:", round(sklearn_score, 2))
    print("Match:", np.isclose(manual_score, sklearn_score))
    print("-" * 30)


# ==========================================
# 6. SIMILARITY MATRIX
# ==========================================

print("\n[5] SIMILARITY MATRIX")

similarity_matrix = sklearn_cosine(local_vectors)

print(np.round(similarity_matrix, 2))


# ==========================================
# 7. TOP 3 MOST SIMILAR PAIRS
# ==========================================

print("\n[6] TOP 3 SIMILAR PAIRS")

pairs = []

for i in range(len(sentences)):

    for j in range(i + 1, len(sentences)):

        score = similarity_matrix[i][j]

        pairs.append((score, i, j))


pairs.sort(reverse=True, key=lambda x: x[0])

for rank, (score, i, j) in enumerate(pairs[:3], start=1):

    print(f"\nRank {rank}")

    print("Sentence 1:", sentences[i])
    print("Sentence 2:", sentences[j])

    print("Similarity:", round(float(score), 4))


# ==========================================
# 8. HIGH SIMILARITY WITHOUT COMMON WORDS
# ==========================================

print("\n[7] SEMANTIC SIMILARITY EXAMPLE")

sentence_a = "The automobile is parked outside."
sentence_b = "A vehicle is standing near the building."

vector_a = model.encode([sentence_a])
vector_b = model.encode([sentence_b])

score = cosine_similarity(
    vector_a[0],
    vector_b[0]
)

print("Sentence 1:", sentence_a)
print("Sentence 2:", sentence_b)
print("Similarity:", round(score, 4))

print("These sentences can have high semantic similarity")
print("even without sharing important common words.")


# ==========================================
# 9. SEMANTIC SEARCH - 3 QUERIES
# ==========================================

print("\n[8] SEMANTIC SEARCH")

queries = [
    "medical coverage for workers",
    "programming languages",
    "weather conditions"
]

for query in queries:

    query_vector = model.encode([query])

    scores = sklearn_cosine(
        query_vector,
        local_vectors
    )[0]

    top_indices = np.argsort(scores)[::-1][:3]

    print("\nQuery:", query)

    for index in top_indices:

        print(
            sentences[index],
            "| Score:",
            round(float(scores[index]), 4)
        )


# ==========================================
# 10. DUPLICATE DETECTION
# ==========================================

print("\n[9] DUPLICATE DETECTION")

threshold = 0.75

print("Threshold:", threshold)

duplicate_found = False

for score, i, j in pairs:

    if score >= threshold:

        duplicate_found = True

        print("\nPossible duplicate:")
        print("Sentence 1:", sentences[i])
        print("Sentence 2:", sentences[j])
        print("Similarity:", round(float(score), 4))

if not duplicate_found:
    print("No duplicate pairs found above threshold.")


# ==========================================
# 11. K-MEANS CLUSTERING
# ==========================================

print("\n[10] K-MEANS CLUSTERING")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(local_vectors)

cluster_names = {
    0: "Technology",
    1: "Weather and Daily Life",
    2: "Health and Insurance"
}

# Name clusters automatically based on their contents
# using the most common semantic category is a simple
# demonstration; the mapping above is illustrative.

for cluster_id in range(3):

    print("\nCluster:", cluster_names[cluster_id])

    for sentence, label in zip(sentences, labels):

        if label == cluster_id:

            print("-", sentence)


# ==========================================
# 12. LOCAL VS API COMPARISON
# ==========================================

print("\n[11] LOCAL VS OPENROUTER COMPARISON")

print("Local model:")
print("Dimensions:", local_vectors.shape[1])
print("Time:", round(local_time, 4), "seconds")

if api_vectors is not None:

    print("\nOpenRouter model:")
    print("Dimensions:", api_vectors.shape[1])
    print("Time:", round(api_time, 4), "seconds")

else:
    print("\nOpenRouter comparison unavailable.")


# ==========================================
# 13. MODEL SELECTION EXPLANATION
# ==========================================

print("""
LOCAL MODEL VS API MODEL

1. Local models are useful for privacy-sensitive data.
2. Local models avoid per-request API charges.
3. API models are convenient and need no local model hosting.
4. Model selection depends on accuracy, latency, cost and scalability.
""")


print("\n" + "=" * 60)
print("TASK 1 COMPLETED")
print("=" * 60)
