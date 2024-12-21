# app.py

import streamlit as st
from typing import List, Dict
from openai import OpenAI
from document_searcher import DocumentSearcher
from config import Config


class QASystem:
    def __init__(self):
        """Initialize QA system with OpenAI client"""
        self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        """Initialize QA system with DocumentSearcher"""
        self.searcher = DocumentSearcher(Config)
    

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            return self.searcher.get_embedding(text)
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise
    

    def search_similar_chunks(self, query: str) -> List[Dict]:
        """Search for similar text chunks"""
        try:
            return self.searcher.search(
                query=query,
                limit=Config.SEARCH_LIMIT,
                score_threshold=Config.SIMILARITY_THRESHOLD
            )
        except Exception as e:
            st.error(f"Error searching database: {str(e)}")
            raise


def initialize_session_state():
    """Initialize session state variables"""
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()


def display_results(results):
    """
    Display search results in a formatted way
    Args:
        results: List of search results
    """
    st.subheader("Search Results")
    
    for i, result in enumerate(results, 1):
        similarity_score = result.score
        background_color = f"rgba(0, 255, 0, {similarity_score})"
        
        with st.expander(
            f"Result {i} (Relevance: {similarity_score:.2%})", 
            expanded=i==1
        ):
            st.markdown(f"""
            📄 **Source Document:** {result.payload['filename']}  
            📝 **Page Number:** {result.payload['page_number']}  
            
            > {result.payload['text']}
            """)


def main():
    # Page configuration
    st.set_page_config(
        page_title="The Mother's Works QA System",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🔍 Question Answering System: The Mother's Works")
    st.markdown("""
    Ask questions about The Mother's works and get relevant passages from the texts.
    """)
    
    # System status check
    try:
        collection_info = st.session_state.qa_system.searcher.get_collection_info()
        vectors_count = collection_info['vectors_count']
        
        if vectors_count == 0:
            st.warning("⚠️ No documents have been processed yet. Please process documents first.")
            return
        
        st.success(f"📚 System ready with {vectors_count:,} indexed text passages")
        
    except Exception as e:
        st.error(f"❌ System Error: Could not connect to the database. {str(e)}")
        return
    
    # Search interface
    with st.form("search_form"):
        user_query = st.text_input(
            "Your Question:",
            placeholder="Enter your question about The Mother's works...",
            help="Type your question and press Enter or click 'Search'"
        )
        
        cols = st.columns([1, 4])
        with cols[0]:
            search_button = st.form_submit_button("🔍 Search")
        
    # Process search
    if search_button and user_query:
        if len(user_query.strip()) < 3:
            st.warning("⚠️ Please enter a longer question")
            return
            
        with st.spinner("🔍 Searching through the texts..."):
            try:
                results = st.session_state.qa_system.search_similar_chunks(user_query)
                
                if results:
                    display_results(results)
                else:
                    st.info("ℹ️ No relevant passages found. Try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"❌ Search Error: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()