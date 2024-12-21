# qdrant_manager.py

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from config import (
    COLLECTION_NAME,
    LOCAL_QDRANT_PATH,
    SEARCH_LIMIT,
    SIMILARITY_THRESHOLD
)

class QdrantClientManager:
    def __init__(self, path: str = LOCAL_QDRANT_PATH):
        """
        Initialize QdrantClientManager
        Args:
            path: Path to local Qdrant storage (defaults to config value)
        """
        self.client = QdrantClient(
            path=path,
            timeout=600.0,  # Default timeout in seconds
            prefer_grpc=True
        )
        self.collection_name = COLLECTION_NAME
        self._create_collection()

    def _create_collection(self) -> None:
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI ada-002 embedding size
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0,
                    )
                )
        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict], texts: List[str]) -> None:
        """
        Add vectors to the collection
        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            texts: List of original texts
        """
        try:
            points = [
                models.PointStruct(
                    id=self._generate_point_id(metadata[i]),
                    vector=vector,
                    payload={
                        'text': texts[i],
                        **metadata[i]
                    }
                )
                for i, vector in enumerate(vectors)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except Exception as e:
            raise Exception(f"Failed to add vectors: {str(e)}")

    def _generate_point_id(self, metadata: Dict) -> int:
        """
        Generate a unique ID for each point based on metadata
        Args:
            metadata: Dictionary containing point metadata
        Returns:
            Unique integer ID
        """
        id_string = f"{metadata['filename']}_{metadata['page_number']}_{metadata['chunk_number']}"
        return abs(hash(id_string)) % (2**63)

    def vectors_exist(self) -> bool:
        """
        Check if vectors exist in the collection
        Returns:
            Boolean indicating if vectors exist
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.vectors_count > 0
        except UnexpectedResponse:
            return False
        except Exception as e:
            raise Exception(f"Failed to check vectors: {str(e)}")

    def search(
        self, 
        query_vector: List[float], 
        limit: int = SEARCH_LIMIT, 
        score_threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Any]:
        """
        Search for similar vectors
        Args:
            query_vector: Query vector embedding
            limit: Maximum number of results (defaults to config value)
            score_threshold: Minimum similarity score (defaults to config value)
        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

    def get_collection_info(self) -> Dict:
        """
        Get information about the collection
        Returns:
            Dictionary containing collection information
        """
        try:
            print("*** Debugging code : reaching get_collection_info method in qdrant_manager")

            info = self.client.get_collection(collection_name=self.collection_name)

            print("*** Debugging code : result received from collection in qdrant_manager", info)

            return {
                'vectors_count': info.points_count,
                'points_count': info.points_count,
                'status': info.status,
                'indexed_vectors_count': info.indexed_vectors_count
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")

    def delete_collection(self) -> None:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")

    def get_scroll_batch(
        self,
        batch_size: int = 100,
        scroll_filter: Dict = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> tuple:
        """
        Get a batch of points using scroll
        Args:
            batch_size: Number of points to retrieve
            scroll_filter: Filter for scrolling
            with_payload: Include payload in results
            with_vectors: Include vectors in results
        Returns:
            Tuple of (points, next_page_offset)
        """
        try:
            return self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
        except Exception as e:
            raise Exception(f"Failed to scroll collection: {str(e)}")