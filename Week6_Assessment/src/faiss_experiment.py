
import time
import numpy as np
import faiss

# ==========================================
# 1. DATA GENERATION
# ==========================================

np.random.seed(42)

N = 50000
DIM = 384
K = 10

vectors = np.random.random(
    (N, DIM)
).astype("float32")

queries = np.random.random(
    (100, DIM)
).astype("float32")

faiss.normalize_L2(vectors)
faiss.normalize_L2(queries)

print("Vectors:", vectors.shape)
print("Queries:", queries.shape)


# ==========================================
# 2. EXACT SEARCH
# ==========================================

exact_index = faiss.IndexFlatIP(DIM)

exact_index.add(vectors)

start = time.perf_counter()

exact_scores, exact_ids = exact_index.search(
    queries, K
)

exact_time = time.perf_counter() - start

print("\nEXACT SEARCH")
print("Time:", round(exact_time * 1000, 2), "ms")


# ==========================================
# 3. HNSW SEARCH
# ==========================================

hnsw_index = faiss.IndexHNSWFlat(
    DIM,
    32,
    faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = 100

hnsw_index.hnsw.efSearch = 128

hnsw_index.add(vectors)

start = time.perf_counter()

hnsw_scores, hnsw_ids = hnsw_index.search(
    queries, K
)

hnsw_time = time.perf_counter() - start

print("\nHNSW SEARCH")
print("Time:", round(hnsw_time * 1000, 2), "ms")


# ==========================================
# 4. RECALL @ 10
# ==========================================

recall_values = []

for exact, approximate in zip(
    exact_ids,
    hnsw_ids
):

    exact_set = set(exact.tolist())

    approximate_set = set(approximate.tolist())

    common = len(exact_set & approximate_set)

    recall = common / K

    recall_values.append(recall)


recall_at_10 = np.mean(recall_values)


# ==========================================
# 5. FINAL RESULTS
# ==========================================

print("\nFINAL COMPARISON")

print(
    "Exact Search:",
    round(exact_time * 1000, 2),
    "ms"
)

print(
    "HNSW Search:",
    round(hnsw_time * 1000, 2),
    "ms"
)

print(
    "Recall@10:",
    round(float(recall_at_10), 4)
)

if hnsw_time > 0:

    print(
        "Speedup:",
        round(exact_time / hnsw_time, 2),
        "x"
    )

print("\nFAISS EXPERIMENT COMPLETED")
