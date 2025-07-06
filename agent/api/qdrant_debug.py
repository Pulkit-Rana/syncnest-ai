# agent/api/qdrant_debug.py

from fastapi import APIRouter
from agent.vector.qdrant_client import client, search_similar
import os

router = APIRouter()

@router.get("/test/qdrant")
async def test_qdrant():
    try:
        collections = client.get_collections()
        return {
            "status": "ok",
            "collections": [c.name for c in collections.collections]
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

@router.get("/test/qdrant/sample")
async def sample_qdrant_docs(limit: int = 5):
    try:
        points = client.scroll(
            collection_name=os.getenv("QDRANT_COLLECTION", "agent-knowledge"),
            limit=limit
        )
        docs = []
        for pt in points[0]:
            docs.append({
                "id": pt.id,
                "payload": pt.payload
            })
        return {
            "status": "ok",
            "count": len(docs),
            "samples": docs
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}

@router.get("/test/qdrant/search")
async def test_semantic_search(query: str, top_k: int = 3):
    try:
        hits = search_similar(query, top_k=top_k)
        return {
            "status": "ok",
            "query": query,
            "hits": hits
        }
    except Exception as e:
        return {"status": "error", "details": str(e)}
