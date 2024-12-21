# Configuration settings

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
api_key = os.environ['OA_API']
os.environ['OPENAI_API_KEY'] = api_key


class Config:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    # Collection settings
    COLLECTION_NAME = "knowledge_base"

    # File paths
    PDF_DIRECTORY = "./data/pdfs/"
    LOCAL_QDRANT_PATH = "./local_qdrant"

    # Search settings
    SEARCH_LIMIT = 10
    SIMILARITY_THRESHOLD = 0.7

    # Text processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    BATCH_SIZE = 8
    TIMEOUT = 600.0