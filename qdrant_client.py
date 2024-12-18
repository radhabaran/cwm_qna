# qdrant_client.py

from qdrant_client import QdrantClient
from config import QDRANT_URL, COLLECTION_NAME

class QdrantClientManager:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = COLLECTION_NAME
        self._create_collection()


    def _create_collection(self):
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vector_size=384,  # Size for the chosen model
                distance='Cosine'
            )


    def add_vectors(self, vectors, metadata):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    'id': i,
                    'vector': vector,
                    'payload': metadata[i]
                }
                for i, vector in enumerate(vectors)
            ]
        )


    def vectors_exist(self):
        return self.client.count(collection_name=self.collection_name) > 0


    def search(self, query_vector, limit=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results