# app.py

import streamlit as st
from typing import List, Dict, Tuple
from openai import OpenAI
from document_searcher import DocumentSearcher
from config import Config


class QASystem:
    def __init__(self):
    
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
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
            f"From '{result.payload['filename']}' (Page {result.payload['page_number']}" + 
            (f" - {result.payload['page_header']}" if result.payload.get('page_header') else "") + 
            f"):\n{result.payload['text']}"
            for result in context
        ])

        messages = [
            {
                "role": "system", 
                "content": """You are a knowledgeable spiritual assistant specialized in 
The Mother's works from Sri Aurobindo Ashram, Pondicherry.

Your task is to answer questions using only the provided context while following these strict rules:

1. Only use conexts where "The Mother" refers specifically to The Mother of Sri Aurobindo Ashram (Mirra Alfassa)
2. Discard any context passages that refer to biological mothers, maternal relationships, or general mother references
3. If a context uses "mother" in lowercase or discusses parent-child relationships, it should be excluded
4. The context must specifically contain The Mother's teachings, writings, or direct quotations from her works
5. If the provided context doesn't contain relevant information from The Mother's works, clearly state that

When citing sources:
1. Start your response with the source citation in a separate line
2. Maintain the original paragraph structure from the source text
3. Add line breaks between paragraphs
4. Present quotes exactly as they appear in the original text
5. When merging multiple passages, separate them clearly with source citations

Remember: "The Mother" in this context exclusively refers to 
Mirra Alfassa (The Mother of Sri Aurobindo Ashram, Pondicherry), who is the author of the 
collected works we are referencing.
"""
            },
            {
                "role": "user",
                "content": f"""Context:
                {context_str}

                Question: {query}"""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
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

    if not results:  # Handle case where all results were filtered out
        st.info("No valid results found after filtering. Try rephrasing your question.")
        return

    if response:
        st.markdown("## AI Response")
        
        # Group results by document and page for primary citation
        primary_result = results[0]
        citation = f"***From {primary_result.payload['filename']}, Page {primary_result.payload['page_number']}"
        if primary_result.payload.get('page_header'):
            citation += f" - {primary_result.payload['page_header']}"
        citation += "***"
        st.markdown(citation)
        st.markdown(f"\"{primary_result.payload['text']}\"")

    if len(results) <= 1:  # No additional results to show
        return

    st.markdown("### Additional Relevant Passages")
    
    # Group results by document
    grouped_results = {}
    for result in results[1:]:  # Skip first result as it's shown above
        doc_key = result.payload['filename']
        if doc_key not in grouped_results:
            grouped_results[doc_key] = []
        grouped_results[doc_key].append(result)

    # Display grouped results
    for doc_name, doc_results in grouped_results.items():
        # Sort results by page number
        doc_results.sort(key=lambda x: x.payload['page_number'])
        
        # Group consecutive pages
        current_group = []
        current_pages = []
        last_page = None
        
        for result in doc_results:
            page_num = result.payload['page_number']
            
            if last_page is not None and page_num != last_page + 1:
                # Display current group
                if current_group:
                    pages_str = f"Page{' ' if len(current_pages) == 1 else 's '}{', '.join(map(str, current_pages))}"

                    # Include page header in expander title if available
                    title = f"***From {doc_name}, {pages_str}"
                    if current_group[0].payload.get('page_header'):
                        title += f" - {current_group[0].payload['page_header']}"
                    title += f"*** [▾ {current_group[0].score:.0%} relevance]"

                    with st.expander(title, expanded=False):
                        for text in current_group:
                            st.markdown(f"\"{text.payload['text']}\"")

                current_group = []
                current_pages = []
            
            current_group.append(result)
            current_pages.append(page_num)
            last_page = page_num
        
        # Display last group
        if current_group:
            pages_str = f"Page{' ' if len(current_pages) == 1 else 's '}{', '.join(map(str, current_pages))}"
            # Include page header in expander title if available
            title = f"***From {doc_name}, {pages_str}"
            if current_group[0].payload.get('page_header'):
                title += f" - {current_group[0].payload['page_header']}"
            title += f"*** [▾ {current_group[0].score:.0%} relevance]"
            
            with st.expander(title, expanded=False):
                for text in current_group:
                    st.markdown(f"\"{text.payload['text']}\"")


def main():
    # Page configuration
    st.set_page_config(
        page_title="QnA bot on the Collected Works of The Mother",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🔍 From the Collected Works of The Mother")
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
        
        # st.success(f"📚 System ready with {vectors_count:,} indexed text passages")
        
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