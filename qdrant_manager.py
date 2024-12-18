# qdrant_manager.py

from qdrant_client import QdrantClient
from config import QDRANT_URL, COLLECTION_NAME

class QdrantClientManager:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, timeout=600)
        self.collection_name = COLLECTION_NAME
        self._create_collection()


    def _create_collection(self):
        # Attempt to get the collection
        collections = self.client.get_collections()
        if self.collection_name not in [collection.name for collection in collections.collections]:
            # Create the collection if it does not exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": 384,  # Size for the chosen model
                    "distance": "Cosine"
                }
            )


    def add_vectors(self, vectors, metadata, texts):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    'id': i,
                    'vector': vector,
                    'payload': {**metadata[i], 'text': texts[i]}  # Include the text in the payload
                }
                for i, vector in enumerate(vectors)
            ]
        )


    def vectors_exist(self):
        count_result = self.client.count(collection_name=self.collection_name)
        return count_result.count > 0  # Access the 'count' attribute


    def search(self, query_vector, limit=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results