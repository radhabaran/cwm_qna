# document_processor.py

import os
from typing import List, Dict, Generator
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm
from config import (
    PDF_DIRECTORY, 
    OPENAI_API_KEY, 
    LOCAL_QDRANT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from qdrant_manager import QdrantClientManager

# Configuration constants
BATCH_SIZE = 8
TIMEOUT = 600.0

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with OpenAI client and QdrantClientManager"""
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=TIMEOUT)
        self.qdrant_manager = QdrantClientManager(path=LOCAL_QDRANT_PATH)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_large_file(self, file_path: str) -> Generator[tuple, None, None]:
        """
        Generator function to process large PDF files
        Args:
            file_path: Path to PDF file
        Yields:
            Tuple of (page_number, page_text)
        """
        reader = pypdf.PdfReader(file_path)
        for page_number, page in enumerate(reader.pages):
            yield page_number, page.extract_text()

    def extract_text_from_pdfs(self, new_files: set) -> List[Dict]:
        """
        Extract text from PDFs using pypdf and configured chunk settings
        Args:
            new_files: Set of new PDF filenames to process
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        text_chunks = []
        
        for filename in os.listdir(PDF_DIRECTORY):
            if filename.endswith('.pdf') and filename in new_files:
                file_path = os.path.join(PDF_DIRECTORY, filename)
                
                for page_number, text in self.process_large_file(file_path):
                    if text:
                        chunks = self.text_splitter.split_text(text)
                        
                        for chunk_number, chunk in enumerate(chunks):
                            text_chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'filename': filename,
                                    'page_number': page_number + 1,
                                    'chunk_number': chunk_number + 1,
                                }
                            })
        
        return text_chunks

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts
        Args:
            texts: List of text strings
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            raise

    def get_processed_files(self) -> set:
        """
        Get set of already processed files
        Returns:
            Set of processed filenames
        """
        try:
            if not self.qdrant_manager.vectors_exist():
                return set()
            
            response = self.qdrant_manager.client.scroll(
                collection_name=self.qdrant_manager.collection_name,
                scroll_filter=None,
                limit=10000,
                with_payload=['filename'],
                with_vectors=False
            )
            return {point.payload['filename'] for point in response[0]}
        except Exception:
            return set()

    def process_documents(self) -> int:
        """
        Process documents with batch operations
        Returns:
            Number of processed chunks
        """
        try:
            # Get processed and new files
            processed_files = self.get_processed_files()
            current_files = {f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')}
            new_files = current_files - processed_files
            
            if not new_files:
                print("No new files to process")
                return 0
            
            # Extract text chunks
            text_chunks = self.extract_text_from_pdfs(new_files)
            
            # Process in batches
            total_batches = (len(text_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Processing {len(text_chunks)} chunks in {total_batches} batches")
            
            for i in tqdm(range(0, len(text_chunks), BATCH_SIZE)):
                batch = text_chunks[i:i + BATCH_SIZE]
                texts = [chunk['text'] for chunk in batch]
                
                # Get embeddings for batch
                embeddings = self.get_embeddings_batch(texts)
                
                # Add vectors to Qdrant
                self.qdrant_manager.add_vectors(
                    vectors=embeddings,
                    metadata=[chunk['metadata'] for chunk in batch],
                    texts=texts
                )
            
            return len(text_chunks)
        
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            raise

def main():
    try:
        processor = DocumentProcessor()
        num_chunks = processor.process_documents()
        print(f"Successfully processed and stored {num_chunks} chunks in Qdrant")
    except Exception as e:
        print(f"Failed to process documents: {str(e)}")

if __name__ == "__main__":
    main()