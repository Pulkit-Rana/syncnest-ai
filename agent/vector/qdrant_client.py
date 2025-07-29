import os
import hashlib
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "agent-knowledge")

#  Embedder setup
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#  Qdrant (embedded mode; for prod server, use url=...)
client = QdrantClient(path="./qdrant_db")

# Create collection (if not exists)
def init_qdrant():
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

#  Safely create Qdrant int ID from any metadata id (int or str)
def _make_int_id(meta_id, fallback_int):
    """
    If meta_id is int-like, use it.
    If string, hash it to int32 for Qdrant embedded compatibility.
    """
    try:
        return int(meta_id)
    except (ValueError, TypeError):
        # Hash string to int32 (first 8 chars of md5 as hex)
        h = hashlib.md5(str(meta_id).encode()).hexdigest()
        return int(h[:8], 16)
    # fallback_int only used if meta_id missing

#  Add documents (no overwrites, no UUID errors)
def add_documents(docs: list[str], metadata_list: list[dict]):
    vectors = model.encode(docs).tolist()
    points = [
        PointStruct(
            id=_make_int_id(metadata_list[i].get('id', i), i),
            vector=vectors[i],
            payload=metadata_list[i]
        )
        for i in range(len(docs))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

#  Query similar documents (semantic search)
def search_similar(text: str, top_k: int = 3):
    query_vector = model.encode(text).tolist()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )
    return [hit.payload for hit in hits]
