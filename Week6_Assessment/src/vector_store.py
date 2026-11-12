
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

from src.chunking import process_all_documents

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

COLLECTION = "company_documents"

MODEL = "openai/text-embedding-3-small"

VECTOR_SIZE = 1536

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

# OpenRouter client
embedding_client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Local persistent Qdrant database
client = QdrantClient(
    path="./qdrant_data"
)


# ==========================================
# 2. EMBEDDING FUNCTION
# ==========================================

def create_embedding(text):

    response = embedding_client.embeddings.create(
        model=MODEL,
        input=text
    )

    return response.data[0].embedding


# ==========================================
# 3. CREATE QDRANT COLLECTION
# ==========================================

def create_collection():

    if not client.collection_exists(COLLECTION):

        client.create_collection(
            collection_name=COLLECTION,

            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

        print("Qdrant collection created!")

    else:

        info = client.get_collection(COLLECTION)

        print("Collection already exists")
        print("Current points:", info.points_count)


# ==========================================
# 4. CREATE PAYLOAD INDEXES
# ==========================================

def create_hnsw_index():

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="doc_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="page",
        field_schema=models.PayloadSchemaType.INTEGER,
        wait=True
    )

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    client.update_collection(
        collection_name=COLLECTION,

        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100
        )
    )

    print("Payload indexes and HNSW configuration created!")


# ==========================================
# 5. INDEX DOCUMENT CHUNKS
# ==========================================

def index_documents():

    chunks = process_all_documents("documents")

    print("\nTotal chunks to index:", len(chunks))

    batch_size = 20

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        points = []

        for chunk in batch:

            text = chunk["text"]

            metadata = chunk["metadata"]

            vector = create_embedding(text)

            # Qdrant point IDs support UUID format.
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["id"]
            ))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": text,
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "section": metadata["section"],
                        "created_date": metadata["created_date"],
                        "chunk_index": metadata["chunk_index"],
                        "doc_type": metadata["doc_type"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True
        )

        print(
            "Indexed:",
            min(start + batch_size, len(chunks)),
            "/",
            len(chunks)
        )

    print("Document indexing completed!")


# ==========================================
# 6. COUNT VECTOR POINTS
# ==========================================

def count_points():

    result = client.count(
        collection_name=COLLECTION,
        exact=True
    )

    return result.count


# ==========================================
# 7. SEARCH DOCUMENTS
# ==========================================

def search_documents(
    query,
    top_k=5,
    doc_type=None,
    min_page=None,
    min_score=None
):

    vector = create_embedding(query)

    conditions = []

    if doc_type:

        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(
                    value=doc_type.upper()
                )
            )
        )

    if min_page is not None:

        conditions.append(
            models.FieldCondition(
                key="page",
                range=models.Range(
                    gte=min_page
                )
            )
        )

    query_filter = None

    if conditions:

        query_filter = models.Filter(
            must=conditions
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True
    ).points

    output = []

    for result in results:

        if (
            min_score is not None
            and result.score < min_score
        ):
            continue

        output.append({
            "source": result.payload["source"],
            "page": result.payload["page"],
            "section": result.payload["section"],
            "score": round(result.score, 4),
            "text": result.payload["text"],
            "doc_type": result.payload["doc_type"]
        })

    return output


# ==========================================
# 8. DELETE DOCUMENT BY SOURCE
# ==========================================

def delete_source(source):

    client.delete(
        collection_name=COLLECTION,

        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(
                            value=source
                        )
                    )
                ]
            )
        ),

        wait=True
    )

    print("Deleted source:", source)


# ==========================================
# 9. MAIN DEMONSTRATION
# ==========================================

if __name__ == "__main__":

    create_collection()

    # Index documents
    index_documents()

    print("\nINITIAL VECTOR COUNT:")
    before = count_points()
    print(before)

    # Filter indexes
    create_hnsw_index()

    # Filtered search
    print("\nFILTERED SEARCH")

    results = search_documents(
        query="employee company policies",
        top_k=5,
        doc_type="TXT",
        min_page=1
    )

    for result in results:

        print("\nSource:", result["source"])
        print("Page:", result["page"])
        print("Score:", result["score"])
        print("Text:", result["text"][:150])

    print("\nFILTERED RESULT COUNT:", len(results))

    # Re-upsert test
    print("\nRE-UPSERT TEST")

    index_documents()

    after = count_points()

    print("Before:", before)
    print("After:", after)

    print("Count unchanged:", before == after)

    # Delete one source
    print("\nDELETE SOURCE TEST")

    source_to_delete = "company_rules.txt"

    delete_source(source_to_delete)

    print("Count after deletion:", count_points())

    print("\nTASK 3 QDRANT COMPLETED")
