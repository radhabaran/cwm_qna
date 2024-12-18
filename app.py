# app.py

import streamlit as st
from document_processor import extract_text_from_pdfs
from vectorizer import Vectorizer
from qdrant_manager import QdrantClientManager

# Initialize components
vectorizer = Vectorizer()
qdrant = QdrantClientManager()

# Check if vectors already exist
if not qdrant.vectors_exist():
    # Load data
    text_chunks = extract_text_from_pdfs('./data/pdfs/')
    
    # Vectorize and store in Qdrant
    texts = [chunk['text'] for chunk in text_chunks]
    metadata = [chunk['metadata'] for chunk in text_chunks]
    vectors = vectorizer.vectorize(texts)
    qdrant.add_vectors(vectors, metadata, texts)
else:
    st.write("Using existing embeddings from Qdrant.")

# Streamlit UI
st.title("Question Answering Bot on the collected works of The Mother")

user_query = st.text_input("Ask a question:")
if st.button("Search"):
    if user_query:
        query_vector = vectorizer.vectorize([user_query])[0]
        results = qdrant.search(query_vector)

        if results:
            for result in results:
                st.write(f"**Source:** {result.payload['filename']}, "
                         f"**Page:** {result.payload['page_number']}, "
                         f"**Text:** {result.payload['text'][:200]}...")  # Display a snippet
        else:
            st.write("No relevant results found.")
    else:
        st.write("Please enter a question.")