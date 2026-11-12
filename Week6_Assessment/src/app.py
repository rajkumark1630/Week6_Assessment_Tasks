
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    UploadFile,
    File
)

from pydantic import BaseModel, Field

from qdrant_client import models

from src.vector_store import (
    client,
    COLLECTION,
    create_embedding
)

from src.chunking import process_document


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("API_KEY")

UPLOAD_FOLDER = Path("documents/uploads")

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}

app = FastAPI(
    title="Company Semantic Search API",
    description="Document semantic search using OpenRouter and Qdrant",
    version="1.0.0"
)


# ==========================================
# 2. API KEY AUTHENTICATION
# ==========================================

def verify_api_key(
    x_api_key: Optional[str] = Header(default=None)
):

    if not API_KEY or x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return True


# ==========================================
# 3. REQUEST MODEL
# ==========================================

class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=2000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# ==========================================
# 4. RESPONSE MODEL
# ==========================================

class SearchResult(BaseModel):

    source: str

    page: int

    section: str

    score: float

    text: str


# ==========================================
# 5. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    try:

        info = client.get_collection(COLLECTION)

        return {
            "status": "healthy",
            "database": "Qdrant",
            "collection": COLLECTION,
            "points_count": info.points_count
        }

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


# ==========================================
# 6. SEARCH FUNCTION
# ==========================================

def search_documents(
    query,
    top_k=5,
    doc_type=None,
    min_score=None
):

    query_vector = create_embedding(query)

    # Fetch extra results so client-side
    # filters do not reduce top_k unnecessarily.

    response = client.query_points(

        collection_name=COLLECTION,

        query=query_vector,

        limit=100,

        with_payload=True

    )

    output = []

    for result in response.points:

        payload = result.payload

        if not payload:
            continue

        if (
            doc_type
            and payload.get("doc_type", "").upper()
            != doc_type.upper()
        ):
            continue

        if (
            min_score is not None
            and result.score < min_score
        ):
            continue

        output.append({

            "source": payload["source"],

            "page": payload["page"],

            "section": payload["section"],

            "score": round(float(result.score), 4),

            "text": payload["text"]

        })

        if len(output) >= top_k:
            break

    return output


# ==========================================
# 7. POST /search
# ==========================================

@app.post(
    "/search",
    response_model=list[SearchResult]
)
def search_api(

    request: SearchRequest,

    doc_type: Optional[str] = None,

    min_score: Optional[float] = None,

    authenticated: bool = Depends(verify_api_key)

):

    try:

        results = search_documents(

            query=request.query,

            top_k=request.top_k,

            doc_type=doc_type,

            min_score=min_score

        )

        return results

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# 8. INDEX UPLOADED DOCUMENT
# ==========================================

def index_uploaded_document(
    filename,
    content
):

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX and TXT files supported"
        )

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(content) > 20 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="File size exceeds 20 MB"
        )

    safe_name = Path(filename).name

    file_path = UPLOAD_FOLDER / safe_name

    file_path.write_bytes(content)

    chunks = process_document(str(file_path))

    if not chunks:

        raise HTTPException(
            status_code=400,
            detail="No extractable text found"
        )

    points = []

    for chunk in chunks:

        metadata = chunk["metadata"]

        vector = create_embedding(
            chunk["text"]
        )

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["id"]
            )
        )

        points.append(

            models.PointStruct(

                id=point_id,

                vector=vector,

                payload={

                    "text": chunk["text"],

                    "source": metadata["source"],

                    "page": metadata["page"],

                    "section": metadata["section"],

                    "created_date": metadata["created_date"],

                    "chunk_index": metadata["chunk_index"],

                    "doc_type": metadata["doc_type"]

                }

            )

        )

    # Remove previous version of this source.
    # This prevents stale chunks remaining after re-upload.

    client.delete(

        collection_name=COLLECTION,

        points_selector=models.FilterSelector(

            filter=models.Filter(

                must=[

                    models.FieldCondition(

                        key="source",

                        match=models.MatchValue(
                            value=safe_name
                        )

                    )

                ]

            )

        ),

        wait=True

    )

    client.upsert(

        collection_name=COLLECTION,

        points=points,

        wait=True

    )

    return {

        "message": "Document uploaded and indexed",

        "source": safe_name,

        "indexed_chunks": len(points)

    }


# ==========================================
# 9. POST /documents
# ==========================================

@app.post("/documents")
async def upload_document(

    file: UploadFile = File(...),

    authenticated: bool = Depends(verify_api_key)

):

    content = await file.read()

    return index_uploaded_document(
        file.filename or "",
        content
    )


# ==========================================
# 10. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
