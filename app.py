#!/usr/bin/env python3
"""
TextMind - Semantic Q&A Interface for The Communist Manifesto

Run this app with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
from typing import Dict, Any, List

# Add the scripts directory to path to allow importing modules
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.append(scripts_dir)

# Import our custom modules
from scripts.question_classifier import classify_question
from scripts.retriever import TextRetriever

# Set page configuration
st.set_page_config(
    page_title="TextMind - The Communist Manifesto Q&A",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add custom CSS
st.markdown("""
<style>
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .answer-type {
        color: #0066cc;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .answer-content {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .source-info {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .question-input {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_retriever() -> TextRetriever:
    """
    Load and initialize the text retriever with caching
    
    Returns:
        Initialized TextRetriever instance
    """
    retriever = TextRetriever()
    
    # Load embeddings if they exist
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "output", "embeddings", "paragraph_embeddings.npy")
    
    if os.path.exists(embeddings_path):
        # Suppress print statements
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Load embeddings and concept dictionary
        retriever.load_embeddings(embeddings_path)
        try:
            retriever.load_concept_dictionary()
        except Exception:
            pass
        
        # Restore stdout
        sys.stdout = old_stdout
    else:
        st.warning("Embeddings not found. Please run scripts/retriever.py first.")
    
    return retriever

# Main app content
def main():
    """Main app function"""
    
    # App header
    st.title("TextMind 🧠")
    st.subheader("Semantic Q&A Interface for The Communist Manifesto")
    st.markdown("""
    Ask any question about Marx and Engels' *The Communist Manifesto*. 
    The system will classify your question, retrieve relevant passages, 
    and highlight key information.
    """)
    
    # Load the retriever
    retriever = load_retriever()
    
    # Question input
    st.markdown("### Ask your question:")
    with st.form(key="question_form"):
        user_question = st.text_input(
            label="Your question",
            placeholder="e.g., What is the relationship between the bourgeoisie and the proletariat?",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(label="Ask")
    
    # Process the question when submitted
    if submit_button and user_question:
        with st.spinner("Analyzing your question..."):
            # Classify the question
            question_type = classify_question(user_question)
            
            # Retrieve relevant passages
            passages = retriever.retrieve_relevant_passages(
                query=user_question,
                top_k=3,
                use_reranking=True
            )
            
            # Display the results
            if passages:
                # Display question type
                st.markdown(f"### Question Type: *{question_type}*")
                
                # Display top result
                top_result = passages[0]
                
                # Create result container
                st.markdown("### Top Answer:")
                result_container = st.container()
                
                with result_container:
                    st.markdown(f"""
                    <div class="answer-box">
                        <div class="answer-content">
                            {top_result['text']}
                        </div>
                        <div class="source-info">
                            Source: Section "{top_result['section']}", Paragraph {top_result['paragraph_index']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show matching concepts if available
                    if 'matching_concepts' in top_result and top_result['matching_concepts']:
                        st.markdown(f"**Matched concepts:** {', '.join(top_result['matching_concepts'])}")
                
                # Display additional results
                if len(passages) > 1:
                    with st.expander("Additional relevant passages"):
                        for i, passage in enumerate(passages[1:], 1):
                            st.markdown(f"""
                            #### Result {i+1}
                            
                            *Section: "{passage['section']}", Paragraph: {passage['paragraph_index']}*
                            
                            {passage['text']}
                            """)
                            
                            if 'matching_concepts' in passage and passage['matching_concepts']:
                                st.markdown(f"**Matched concepts:** {', '.join(passage['matching_concepts'])}")
                            
                            st.markdown("---")
            else:
                st.error("No relevant passages found. Please try rephrasing your question.")
    
    # Sample questions
    with st.expander("Sample questions to try"):
        st.markdown("""
        - What is the bourgeoisie?
        - How does Marx define communism?
        - When was the Communist Manifesto written?
        - Why does Marx believe capitalism will collapse?
        - What examples does Marx provide for worker exploitation?
        - How does Marx compare capitalism to feudalism?
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*TextMind project - Semantic Q&A for historical texts*")

if __name__ == "__main__":
    main() 