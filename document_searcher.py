# document_searcher.py

from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import re


class DocumentSearcher:
    def __init__(self, config):
        """Initialize searcher with OpenAI client and Qdrant client"""
        self.config = config
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
        self.qdrant_client = QdrantClient(path=config.LOCAL_QDRANT_PATH)


    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for search query"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding
    

    def clean_text(self, text: str) -> str:
        """Clean text by removing trailing numbers and extra whitespace"""
        # Remove trailing numbers and clean whitespace
        text = text.strip()
        text = re.sub(r'\s*\d+\s*$', '', text)
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text


    def is_valid_content(self, text: str, query: str) -> bool:
        """Filter out metadata, headers, and navigational content"""

        # Only apply Mother-specific filtering if the query is about The Mother
        if ('mother' in query.lower() or 'The Mother' in query) and 'mother' in text.lower():
            if 'The Mother' not in text:
                return False

        if len(text.strip()) < 50:  # Minimum 50 characters
            return False
            
        metadata_patterns = [
            r"^The Mother taking a class",
            r"^Page \d+$",
            r"^Chapter \d+$",
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            r"^Table of Contents$",
            r"^Questions and Answers$",
            r"^[\d\s\-—]*$",  # Just numbers, spaces, dashes
            r"^\s*\(.*\)\s*$"  # Just parenthetical content
        ]
        
        # Skip metadata pattern checking if this is the page header
        if isinstance(text, dict) and 'page_header' in text:
            return True

        for pattern in metadata_patterns:
            if re.match(pattern, text.strip()):
                return False
                
        return True

        
    def search(self, query: str, limit: int = None, score_threshold: float = None) -> List[Dict]:
        """Search for similar text chunks"""
        query_vector = self.get_embedding(query)
        
        results = self.qdrant_client.search(
            collection_name=self.config.COLLECTION_NAME,
            query_vector=query_vector,
            # limit=limit or self.config.SEARCH_LIMIT,
            limit=(limit or self.config.SEARCH_LIMIT) * 2,  # Get more results to account for filtering
            score_threshold=score_threshold or self.config.SIMILARITY_THRESHOLD
        )

        # Filter out invalid content and clean text
        filtered_results = []
        for result in results:
            if self.is_valid_content(result.payload['text'], query):
                # Clean and format the text
                text = self.clean_text(result.payload['text'])
            
                # Format the result as a dictionary
                clean_result = {
                    'filename': result.payload['filename'],
                    'page_number': result.payload['page_number'],
                    'score': result.score
                }
                # Build the formatted result
                formatted_parts = []

                # Format citation and text
                citation = f"From {result.payload['filename']}, Page {result.payload['page_number']}\n\n"
                formatted_parts.append(citation)

                # Add header if available
                if result.payload.get('page_header'):
                    header = result.payload['page_header'].strip().strip('"')
                    # Remove header from text if it appears at the start
                    if text.startswith(header):
                        text = text[len(header):].strip()
                    formatted_parts.append(header)

                # Add main text
                formatted_parts.append(text)
                
                # Join all parts with double newlines and store in result
                clean_result['text'] = '\n\n'.join(part for part in formatted_parts if part)
            
                filtered_results.append(clean_result)
                
        # Return only up to the requested limit
        return filtered_results[:limit or self.config.SEARCH_LIMIT]


    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        info = self.qdrant_client.get_collection(
            collection_name=self.config.COLLECTION_NAME
        )
        return {
            'vectors_count': info.points_count,
            'points_count': info.points_count,
            'status': info.status,
            'indexed_vectors_count': info.indexed_vectors_count
        }