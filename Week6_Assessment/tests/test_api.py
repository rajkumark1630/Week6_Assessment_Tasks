
import pytest

from fastapi.testclient import TestClient

from src.app import app


# ==========================================
# CONFIGURATION
# ==========================================

API_KEY = "week6-secret-123"

HEADERS = {
    "X-API-Key": API_KEY
}


@pytest.fixture
def client():

    with TestClient(app) as test_client:
        yield test_client


# ==========================================
# TEST 1: HEALTH CHECK
# ==========================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["database"] == "Qdrant"

    assert "points_count" in data


# ==========================================
# TEST 2: API KEY PROTECTION
# ==========================================

def test_missing_api_key(client):

    response = client.post(
        "/search",
        json={
            "query": "employee policy",
            "top_k": 5
        }
    )

    assert response.status_code == 401

    assert response.json()["detail"] == "Invalid or missing API key"


# ==========================================
# TEST 3: INVALID TOP_K
# ==========================================

def test_invalid_top_k(client):

    response = client.post(
        "/search",
        headers=HEADERS,
        json={
            "query": "employee policy",
            "top_k": 25
        }
    )

    assert response.status_code == 422


# ==========================================
# TEST 4: SEARCH SUCCESS
# ==========================================

def test_search_success(client, monkeypatch):

    def fake_search(
        query,
        top_k=5,
        doc_type=None,
        min_score=None
    ):

        return [
            {
                "source": "employee_policy.txt",
                "page": 1,
                "section": "EMPLOYEE POLICY",
                "score": 0.85,
                "text": "Employee health insurance benefits"
            }
        ]

    monkeypatch.setattr(
        "src.app.search_documents",
        fake_search
    )

    response = client.post(
        "/search",
        headers=HEADERS,
        json={
            "query": "employee health benefits",
            "top_k": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1

    assert data[0]["source"] == "employee_policy.txt"

    assert data[0]["score"] == 0.85


# ==========================================
# TEST 5: SEARCH FILTERS
# ==========================================

def test_search_filters(client, monkeypatch):

    captured = {}

    def fake_search(
        query,
        top_k=5,
        doc_type=None,
        min_score=None
    ):

        captured["query"] = query
        captured["top_k"] = top_k
        captured["doc_type"] = doc_type
        captured["min_score"] = min_score

        return []

    monkeypatch.setattr(
        "src.app.search_documents",
        fake_search
    )

    response = client.post(
        "/search?doc_type=TXT&min_score=0.5",
        headers=HEADERS,
        json={
            "query": "security policies",
            "top_k": 5
        }
    )

    assert response.status_code == 200

    assert captured["doc_type"] == "TXT"

    assert captured["min_score"] == 0.5

    assert captured["top_k"] == 5


# ==========================================
# TEST 6: DOCUMENT UPLOAD
# ==========================================

def test_document_upload(client, monkeypatch):

    def fake_index(filename, content):

        return {
            "message": "Document uploaded and indexed",
            "source": filename,
            "indexed_chunks": 2
        }

    monkeypatch.setattr(
        "src.app.index_uploaded_document",
        fake_index
    )

    response = client.post(
        "/documents",
        headers=HEADERS,
        files={
            "file": (
                "test_document.txt",
                b"Company employee policies",
                "text/plain"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["source"] == "test_document.txt"

    assert data["indexed_chunks"] == 2

    assert data["message"] == "Document uploaded and indexed"


# ==========================================
# TESTS COMPLETED
# ==========================================
