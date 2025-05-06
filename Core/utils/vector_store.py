import torch
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from utils.custom_embedding import M3EmbeddingsAPI



QDRANT_URL = "https://291e3dc4-2f58-4a51-9795-3be0f4d2ae1d.us-east4-0.gcp.cloud.qdrant.io"

if os.path.exists('qdrant_read_key.txt'):
    with open("qdrant_read_key.txt", "r", encoding="utf-8") as file:
        QDRANT_API_KEY = file.read()
else:
    QDRANT_API_KEY = ""

COLLECTION_NAME = "law_collection_backup"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", None)

def init_vector_store():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bge_m3_emb_size = 1024

    model_name = "BAAI/bge-m3"
    model_kwargs = {"torch_dtype": torch.float16, "device": device}
    encode_kwargs = {"batch_size": 8, "normalize_embeddings": True}

    if NVIDIA_API_KEY:
        bge_m3_emb = M3EmbeddingsAPI(nvidia_api_key=NVIDIA_API_KEY)
    else:
        bge_m3_emb = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="",
        )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]
    if COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=bge_m3_emb_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=bge_m3_emb,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    return vector_store


