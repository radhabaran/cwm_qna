# vectorizer.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def vectorize(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()