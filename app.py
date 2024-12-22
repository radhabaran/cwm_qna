# app.py

import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from document_searcher import DocumentSearcher
from config import Config


class QASystem:
    def __init__(self):
        """Initialize QA system with OpenAI client"""
        self.model = OpenAI(api_key=Config.OPENAI_API_KEY, model="gpt-3.5-turbo")
        """Initialize QA system with DocumentSearcher"""
        self.searcher = DocumentSearcher(Config)
    

    def get_chat_response(self, query: str, context: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Get response from GPT-3.5 Turbo using context
        Args:
            query: User's question
            context: List of relevant text chunks with metadata
        Returns:
            Tuple of (response text, context used)
        """
        # Prepare context string
        context_str = "\n\n".join([
            f"From '{result.payload['filename']}' (Page {result.payload['page_number']}):\n{result.payload['text']}"
            for result in context
        ])

        # Prepare the prompt
        prompt = f"""You are a knowledgeable spiritual assistant helping users understand The Mother's
collected works. Here the Mother is the Mother from Sri Aurobindo Ashram from Pondicherry. Use the following
context to answer the question. If you cannot answer the question based on the context, say so. 
Always cite the source document and page number when providing information.

Context:
{context_str}

Question: {query}

Please provide a clear and concise answer based on the above context."""

        messages = [
            {"role": "system", "content": "You are a knowledgeable spiritual assistant specialized in The Mother's works."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.model.invoke(
                input=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content, context
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            raise


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


def display_results(results, response=None):
    """
    Display search results in a formatted way
    Args:
        results: List of search results
        response: Optional AI response to display
    """
    if response:
        st.subheader("AI Response")
        st.write(response)

    st.subheader("Supporting Passages")
    
    for i, result in enumerate(results, 1):
        similarity_score = result.score
        
        with st.expander(
            f"Source {i} (Relevance: {similarity_score:.2%})", 
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
        page_title="QnA bot on the Collected Works of The Mother",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🔍 Fromm the Collected Works of The Mother")
    st.markdown("""
    Ask questions about The Mother's works and get AI-powered answers with relevant passages.
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
            
        with st.spinner("🔍 Searching and generating response..."):
            try:
                results = st.session_state.qa_system.search_similar_chunks(user_query)
                
                if results:
                    response, context = st.session_state.qa_system.get_chat_response(user_query, results)
                    display_results(results, response)
                else:
                    st.info("ℹ️ No relevant passages found. Try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"❌ Search Error: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")


if __name__ == "__main__":
    main()