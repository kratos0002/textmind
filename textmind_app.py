#!/usr/bin/env python3
"""
TextMind - Interactive Q&A Interface for The Communist Manifesto with User Feedback

Run this app with:
    streamlit run textmind_app.py
"""

import os
import sys
import streamlit as st
from typing import Dict, Any, List, Optional
import json
import time
import datetime
import logging
import re
import uuid
import yaml
import html

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "textmind_app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the scripts directory to path to allow importing modules
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
sys.path.append(scripts_dir)

# Import required modules
from scripts.retriever import TextRetriever
from scripts.reranker import rerank_passages, explain_ranking
from scripts.question_classifier import classify_question
from scripts.answer_generator import AnswerGenerator, OLLAMA_AVAILABLE
from feedback_store import feedback_store
from session_manager import session_manager
from feedback_logger import log_feedback
from refine_answer import refine_answer

# Load configuration if available
CONFIG_FILE = "config.yaml"
config = {}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Error loading configuration: {str(e)}")
        config = {}
else:
    logger.info(f"Configuration file {CONFIG_FILE} not found, using defaults")

# Apply configuration with defaults
ENABLE_FEEDBACK_LOGGING = config.get("enable_feedback_logging", True)
ENABLE_SESSION_MANAGEMENT = config.get("enable_session_management", True)
ENABLE_DEBUG_MODE = config.get("enable_debug_mode", False)
DEFAULT_MODEL = config.get("default_model", "ollama")
DEFAULT_OLLAMA_MODEL = config.get("default_ollama_model", "mistral")
MAX_ANSWER_LENGTH = config.get("max_answer_length", 512)
MAX_PASSAGES = config.get("max_passages_to_retrieve", 5)
DATA_FILE = config.get("data_file", "data/manifesto_new_with_concepts.jsonl")
EMBEDDINGS_DIR = config.get("embeddings_dir", "output/embeddings_new")
LOGS_DIR = config.get("logs_dir", "logs")
FEEDBACK_LOG_FILE = config.get("feedback_log_file", "logs/feedback_log.csv")

# Set page configuration
st.set_page_config(
    page_title="TextMind - Ask Marx Anything",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add custom CSS
st.markdown("""
<style>
    /* Main container styles */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Header styles */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Question input styles */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 10px;
    }
    
    /* Card styles */
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #eaecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Answer styles */
    .answer-card {
        background-color: #f0f7ff;
        border-left: 4px solid #0066cc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0;
    }
    
    .answer-text {
        font-size: 1.2rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Question tag styles */
    .question-tag {
        display: inline-block;
        background-color: #e6f3ff;
        color: #0066cc;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    /* Source passage styles */
    .source-passage {
        border-left: 3px solid #4caf50;
        padding-left: 1rem;
        margin: 1rem 0;
        color: #555;
        font-style: italic;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Passage text truncation */
    .truncated-text {
        position: relative;
    }
    
    .read-more {
        color: #0066cc;
        cursor: pointer;
        font-weight: bold;
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    /* Feedback button styles */
    .feedback-btn {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin-right: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #ddd;
    }
    
    .thumbs-up {
        background-color: #e6f7e6;
        color: #4caf50;
    }
    
    .thumbs-down {
        background-color: #ffebee;
        color: #f44336;
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Conversation history styles */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #f0f7ff;
        border-left: 3px solid #0066cc;
    }
    
    .system-message {
        background-color: #f9f9f9;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load the retriever
@st.cache_resource
def load_retriever(data_file="data/manifesto_new_with_concepts.jsonl", output_dir="output/embeddings_new"):
    """Load the text retriever with embeddings"""
    logger.info(f"Initializing retriever with data from {data_file}")
    
    # Directory setup
    embeddings_path = os.path.join(output_dir, "paragraph_embeddings.npy")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the retriever
    retriever = TextRetriever()
    
    # Load paragraphs
    retriever.load_paragraphs(jsonl_path=data_file)
    
    # Ensure embeddings are loaded/created
    try:
        retriever.load_embeddings(embeddings_path=embeddings_path)
        logger.info(f"Successfully loaded embeddings for {len(retriever.paragraphs)} paragraphs")
    except FileNotFoundError:
        st.warning(f"Embeddings not found at {embeddings_path}")
        st.warning("Creating embeddings for the first time might take a minute...")
        retriever.create_embeddings()
        retriever.save_embeddings(embeddings_path=embeddings_path)
        logger.info("Created and saved new embeddings")
    
    return retriever

# Initialize answer generator
@st.cache_resource
def load_answer_generator(use_ollama=True, ollama_model="mistral"):
    """Initialize the answer generator with appropriate settings"""
    # Only use Transformers if explicitly requested and Ollama not available
    use_transformers = not (use_ollama and OLLAMA_AVAILABLE)
    
    generator = AnswerGenerator(
        use_llm=use_transformers,  # Use Transformers only if Ollama not used 
        model_name="gpt2",         # Default Transformers model
        use_ollama=use_ollama,     # Prefer Ollama if available
        ollama_model=ollama_model  # Ollama model to use
    )
    
    if use_ollama and OLLAMA_AVAILABLE:
        logger.info(f"Using Ollama with model {ollama_model}")
    elif use_transformers:
        logger.info("Using Transformers for answer generation")
    else:
        logger.info("Using rule-based answer generation")
        
    return generator

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    Highlight keywords in the text using HTML
    
    Args:
        text: The text to highlight
        keywords: List of keywords to highlight
        
    Returns:
        Text with HTML highlighting for keywords
    """
    if not keywords:
        return text
    
    # Create a copy of the text for highlighting
    highlighted_text = text
    
    for keyword in keywords:
        if not keyword:
            continue
            
        # Case insensitive replacement with regex
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        replacement = f'<span class="matched-keyword">{keyword}</span>'
        highlighted_text = pattern.sub(replacement, highlighted_text)
    
    return highlighted_text

def strip_html_tags(text):
    """
    Remove all HTML-like tags from the text
    
    Args:
        text: Text that might contain HTML tags
        
    Returns:
        Text with all HTML tags removed
    """
    if not text:
        return ""
    
    # Use a more aggressive regex pattern to remove HTML-like tags
    clean_text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    return clean_text

def clean_answer_text(response):
    """
    Clean any HTML tags from the answer text
    
    Args:
        response: The response dictionary with answer text
        
    Returns:
        Modified response with cleaned answer text
    """
    if not response or 'answer' not in response:
        return response
    
    # Make a copy of the response
    cleaned_response = dict(response)
    
    # Clean the answer text
    cleaned_response['answer'] = strip_html_tags(cleaned_response['answer'])
    
    return cleaned_response

def process_question(
    question: str,
    retriever,
    answer_generator,
    session,
    use_ollama=True,
    ollama_model="mistral",
    use_transformers=False,
    use_reranker=True,
    top_n=5,
    is_follow_up=False,
    debug=False
):
    """
    Process a question through the TextMind pipeline
    
    Args:
        question: The user's question
        retriever: TextRetriever instance
        answer_generator: AnswerGenerator instance
        session: Session object for conversation history
        use_ollama: Whether to use Ollama for generation
        ollama_model: Ollama model name
        use_transformers: Whether to use Transformers for generation
        use_reranker: Whether to use reranking
        top_n: Number of passages to retrieve
        is_follow_up: Whether this is a follow-up question
        debug: Whether to show debug info
        
    Returns:
        Dictionary with response data
    """
    # Record start time
    start_time = datetime.datetime.now()
    
    # Get conversation context for follow-up questions
    conversation_context = None
    previous_interaction = None
    if is_follow_up and session:
        conversation_context = session.get_recent_context(max_interactions=2)
        
        # Get the most recent interaction for refinement
        if session.history:
            previous_interaction = session.history[-1]
    
    # Handle follow-up question through direct refinement if possible
    if is_follow_up and previous_interaction:
        try:
            # Use refine_answer to get a refined answer based on the previous interaction
            refined_answer = refine_answer(
                original_question=previous_interaction.get('question', ''),
                original_answer=previous_interaction.get('answer', ''),
                follow_up_question=question,
                model="ollama" if use_ollama else "rule",
                ollama_model=ollama_model
            )
            
            # Clean the refined answer
            refined_answer = strip_html_tags(refined_answer)
            
            # Prepare refinement response
            response = {
                "question": question,
                "question_type": previous_interaction.get('question_type', 'Context'),
                "confidence": 0.8,  # Default confidence for refinements
                "answer": refined_answer,
                "answer_confidence": 0.8,
                "generation_method": "refinement",
                "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
                "is_follow_up": True,
                "session_id": session.session_id if session else None,
                "passage": previous_interaction.get('metadata', {}).get('passage', None),
                "passage_info": {
                    "id": previous_interaction.get('metadata', {}).get('passage_id', ''),
                    "text": previous_interaction.get('metadata', {}).get('passage_text', ''),
                    "hybrid_score": 0.0,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "matched_keywords": []
                },
                "all_passages": [],
                "ranking_explanation": None,
                "classification": {
                    "question_type": previous_interaction.get('question_type', 'Context'),
                    "confidence": 0.8
                },
                "debug": {
                    "conversation_context": conversation_context,
                    "refinement": True,
                    "original_question": previous_interaction.get('question', ''),
                    "original_answer": previous_interaction.get('answer', '')
                }
            }
            
            # Add to session history
            if session:
                metadata = {
                    "question_type": previous_interaction.get('question_type', 'Context'),
                    "method": "refinement",
                    "passage_id": previous_interaction.get('metadata', {}).get('passage_id', ''),
                    "passage_text": previous_interaction.get('metadata', {}).get('passage_text', ''),
                    "is_follow_up": True,
                    "refinement": True,
                    "original_question": previous_interaction.get('question', ''),
                    "execution_time": response["execution_time"]
                }
                session.add_interaction(
                    question=question,
                    answer=refined_answer,
                    question_type=previous_interaction.get('question_type', 'Context'),
                    metadata=metadata
                )
            
            return response
        except Exception as e:
            logger.warning(f"Refinement failed, falling back to standard processing: {str(e)}")
            # Continue with regular processing if refinement fails
    
    # Step 1: Classify the question
    classification = classify_question(question)
    
    # Handle different return formats from classify_question
    if isinstance(classification, dict):
        question_type = classification.get("category", "Unknown")
        confidence = classification.get("confidence_scores", {}).get(question_type, 0.0)
    else:
        # Backward compatibility with older classifier
        question_type = classification
        confidence = 0.8  # Default confidence
        classification = {
            "question_type": question_type,
            "confidence": confidence
        }
    
    # Augment query with conversation context for follow-up questions
    augmented_query = question
    if is_follow_up and conversation_context:
        logger.info("Processing as follow-up question with context")
        augmented_query = f"{conversation_context}\nFollow-up: {question}"
        if debug:
            logger.info(f"Augmented query: {augmented_query}")
    
    # Step 2: Retrieve relevant passages
    passages = retriever.retrieve_relevant_passages(
        augmented_query if is_follow_up else question, 
        top_k=top_n
    )
    
    # Step 3: Apply reranking if enabled
    if use_reranker and passages:
        reranked_passages = rerank_passages(
            augmented_query if is_follow_up else question, 
            passages, 
            question_type
        )
        ranking_explanation = explain_ranking(reranked_passages)
    else:
        reranked_passages = passages
        ranking_explanation = None
    
    # Step 4: Get the top passage
    top_passage = reranked_passages[0] if reranked_passages else None
    
    # Step 5: Generate answer
    if not top_passage:
        answer_data = {
            "answer": "I couldn't find a relevant passage to answer your question.",
            "confidence": 0.0,
            "method": "fallback"
        }
    else:
        # Determine generation method based on settings
        if use_ollama and OLLAMA_AVAILABLE:
            method = "ollama"
        elif use_transformers:
            method = "llm"
        else:
            method = "rule"
            
        # Generate answer
        answer_data = answer_generator.generate_answer(
            question=question,
            passage=top_passage,
            question_type=question_type,
            method=method
        )
    
    # Calculate execution time
    execution_time = (datetime.datetime.now() - start_time).total_seconds()
    
    # Add to session history
    if session:
        metadata = {
            "question_type": question_type,
            "method": answer_data.get("method", ""),
            "passage_id": top_passage.get("paragraph_id", "") if top_passage else None,
            "passage_text": top_passage.get("text", "") if top_passage else None,
            "is_follow_up": is_follow_up,
            "execution_time": execution_time
        }
        session.add_interaction(
            question=question,
            answer=answer_data.get("answer", ""),
            question_type=question_type,
            metadata=metadata
        )
    
    # Prepare response
    response = {
        "question": question,
        "question_type": question_type,
        "confidence": confidence,
        "answer": answer_data.get("answer", ""),
        "answer_confidence": answer_data.get("confidence", 0.0),
        "generation_method": answer_data.get("method", ""),
        "execution_time": execution_time,
        "is_follow_up": is_follow_up,
        "session_id": session.session_id if session else None,
        "passage": top_passage,
        "passage_info": {
            "id": top_passage.get("paragraph_id", "") if top_passage else None,
            "text": top_passage.get("text", "") if top_passage else None,
            "hybrid_score": top_passage.get("hybrid_score", 0.0) if top_passage else 0.0,
            "semantic_score": top_passage.get("score", 0.0) if top_passage else 0.0,
            "keyword_score": top_passage.get("keyword_score", 0.0) if top_passage else 0.0,
            "matched_keywords": top_passage.get("matched_keywords", []) if top_passage else []
        },
        "all_passages": reranked_passages[:top_n] if reranked_passages else [],
        "ranking_explanation": ranking_explanation,
        "classification": classification,
        "debug": {
            "conversation_context": conversation_context,
            "augmented_query": augmented_query if is_follow_up else None,
            "retrieved_passages_count": len(passages),
            "reranked_passages_count": len(reranked_passages) if reranked_passages else 0
        }
    }
    
    # Clean the response before returning
    return clean_answer_text(response)

def display_conversation_history(session):
    """
    Display the conversation history
    
    Args:
        session: Session object with conversation history
    """
    if not session or not session.history:
        return
    
    history = session.get_history()
    
    for i, interaction in enumerate(history):
        # Display user question
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>Q:</strong> {interaction['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display system answer
        st.markdown(f"""
        <div class="chat-message system-message">
            <strong>A:</strong> {interaction['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        if i < len(history) - 1:
            st.markdown("<hr style='margin: 0.5rem 0; border-color: #eee;'>", unsafe_allow_html=True)

# Replace the display_card and display_followup_card functions with direct Streamlit components
def display_card(response):
    """
    Display the question and answer using Streamlit components instead of HTML
    
    Args:
        response: Response data containing question, answer, etc.
    """
    # Don't use HTML for display - use direct Streamlit components
    st.markdown(f"**Question:** {response['question']}")
    
    # Display the question type tag
    st.markdown(f"<span class='question-tag'>{response['question_type']}</span>", unsafe_allow_html=True)
    
    # Display the answer in a styled container
    with st.container():
        st.markdown("### Answer:")
        st.write(response['answer'])
    
    # Display the source
    st.markdown("**Source:**")
    
    # Display the passage in a scrollable area
    if response["passage_info"]["text"]:
        st.markdown(f"<div class='source-passage'>{response['passage_info']['text']}</div>", unsafe_allow_html=True)
        st.caption("Scroll to see more")
    else:
        st.info("No specific source passage found.")


def display_followup_card(response):
    """
    Display a follow-up question and answer using Streamlit components
    
    Args:
        response: Response data for the follow-up
    """
    # Don't use HTML for display - use direct Streamlit components
    st.markdown(f"**Follow-up Question:** {response['question']}")
    
    # Display the question type tag
    st.markdown(f"<span class='question-tag'>{response['question_type']}</span>", unsafe_allow_html=True)
    
    # Display the answer in a styled container
    with st.container():
        st.markdown("### Answer:")
        st.write(response['answer'])
    
    # Display the source
    st.markdown("**Source:**")
    
    # Display the passage in a scrollable area
    if response["passage_info"]["text"]:
        st.markdown(f"<div class='source-passage'>{response['passage_info']['text']}</div>", unsafe_allow_html=True)
        st.caption("Scroll to see more")
    else:
        st.info("No specific source passage found.")

def main():
    """Main app function"""
    
    # Initialize resources
    retriever = load_retriever(data_file=DATA_FILE, output_dir=EMBEDDINGS_DIR)
    
    # Initialize or retrieve session
    if "session_id" not in st.session_state:
        session = session_manager.create_session()
        st.session_state.session_id = session.session_id
    else:
        session = session_manager.get_session(st.session_state.session_id)
        if not session:
            # Session expired or not found, create a new one
            session = session_manager.create_session()
            st.session_state.session_id = session.session_id
    
    # Initialize response state
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
        
    # Initialize feedback state
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    
    # Main header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="header">
            <h1>🧠 TextMind: Ask Marx Anything</h1>
        </div>
        """, unsafe_allow_html=True)

    # App introduction
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Explore the ideas from <em>The Communist Manifesto</em> through contextual questions and answers.
    </div>
    """, unsafe_allow_html=True)
    
    # Generation style selector
    gen_method_col1, gen_method_col2 = st.columns([3, 1])
    with gen_method_col1:
        generation_style = st.radio(
            "Choose your answer style:",
            options=["Smart & Balanced", "Simple & Direct", "Creative & Explorative"],
            index=0,
            horizontal=True
        )
    
    # Map user-friendly terms to technical methods
    if generation_style == "Smart & Balanced":
        use_ollama = True
        use_transformers = False
        ollama_model = DEFAULT_OLLAMA_MODEL
    elif generation_style == "Simple & Direct":
        use_ollama = False
        use_transformers = False
        ollama_model = DEFAULT_OLLAMA_MODEL
    else:  # Creative & Explorative
        use_ollama = False
        use_transformers = True
        ollama_model = DEFAULT_OLLAMA_MODEL
    
    # Initialize answer generator with selected method
    answer_generator = load_answer_generator(
        use_ollama=use_ollama,
        ollama_model=ollama_model
    )
    
    # Question input section
    st.markdown("<div style='margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
    
    user_question = st.text_input(
        "Ask a question",
        placeholder="Ask Marx a question...",
        label_visibility="collapsed"
    )
    
    st.markdown(
        "<div style='text-align: center; font-size: 0.9rem; color: #666; margin-top: -0.5rem;'>"
        "<em>We'll find the best answer from The Communist Manifesto.</em>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Add a checkbox to mark as follow-up if there's a conversation history
    is_follow_up = False
    if session and session.history:
        is_follow_up = st.checkbox(
            "This is a follow-up to my previous question",
            value=True if len(session.history) > 0 else False
        )
    
    # Ask button
    ask_col1, ask_col2, ask_col3 = st.columns([1, 1, 1])
    with ask_col2:
        ask_button = st.button("Ask", use_container_width=True, type="primary")
    
    # Advanced settings (collapsible)
    with st.expander("Advanced Settings", expanded=False):
        use_reranker = st.checkbox("Use hybrid reranking", value=True)
        top_n = st.slider("Number of passages to retrieve", min_value=1, max_value=10, value=5)
        
        # Show Ollama status if user wants to use it
        if use_ollama:
            if OLLAMA_AVAILABLE:
                st.success("✅ Ollama is available")
            else:
                st.error("❌ Ollama is not available")
                st.info("Falling back to rule-based generation")
        
        # Show details about each generation style
        st.markdown("""
        #### Answer Generation Styles:
        - **Smart & Balanced**: Uses Ollama LLM for nuanced, contextual answers (best quality).
        - **Simple & Direct**: Uses rule-based templates for straightforward, factual answers (fastest).
        - **Creative & Explorative**: Uses transformer models for more varied responses (experimental).
        """)
        
        # Debug toggle (only if enabled in config)
        show_debug = False
        if ENABLE_DEBUG_MODE:
            show_debug = st.checkbox("Show debug information", value=False)
    
    # Display conversation history in expandable section
    if session and session.history:
        with st.expander("View Conversation History", expanded=False):
            display_conversation_history(session)
    
    # Process question when asked
    if ask_button and user_question:
        # Reset feedback state
        st.session_state.feedback_submitted = False
        
        with st.spinner("Thinking..."):
            # Process the question
            response = process_question(
                question=user_question,
                retriever=retriever,
                answer_generator=answer_generator,
                session=session,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                use_transformers=use_transformers,
                use_reranker=use_reranker,
                top_n=top_n,
                is_follow_up=is_follow_up,
                debug=ENABLE_DEBUG_MODE and show_debug
            )
            
            # Clean the response before storing
            response = clean_answer_text(response)
            st.session_state.last_response = response
        
        # Divider
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Display the question and answer in a card format
        display_card(response)
        
        # Add feedback buttons
        st.markdown("<div style='margin-top: 1rem;'><strong>Was this answer helpful?</strong></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 5])
        
        with col1:
            thumbs_up = st.button("👍 Helpful")
        
        with col2:
            thumbs_down = st.button("👎 Not Helpful")
        
        # Process feedback
        feedback_comment = ""
        if thumbs_up or thumbs_down:
            feedback_type = "thumbs_up" if thumbs_up else "thumbs_down"
            
            # If thumbs down, show comment field
            if thumbs_down:
                feedback_comment = st.text_area("How can we improve this answer?", "", key="feedback_comment")
            
            # Prepare feedback data
            feedback_data = {
                "question": response["question"],
                "answer": response["answer"],
                "passage": response["passage_info"]["text"] if response["passage_info"]["text"] else "",
                "model_used": f"{response['generation_method']}-{ollama_model}" if response['generation_method'] == "ollama" else response['generation_method'],
                "feedback_type": feedback_type,
                "user_comment": feedback_comment,
                "follow_up_question": None,  # Will be updated if a follow-up is submitted
                "metadata": {
                    "question_type": response["question_type"],
                    "passage_id": response["passage_info"]["id"],
                    "semantic_score": response["passage_info"]["semantic_score"],
                    "keyword_score": response["passage_info"]["keyword_score"],
                    "hybrid_score": response["passage_info"]["hybrid_score"],
                    "is_follow_up": response["is_follow_up"],
                    "session_id": session.session_id if session else None,
                    "execution_time": response["execution_time"]
                }
            }
            
            # Store feedback
            if ENABLE_FEEDBACK_LOGGING:
                log_feedback(
                    question=feedback_data["question"],
                    answer=feedback_data["answer"],
                    passage=feedback_data["passage"],
                    model_used=feedback_data["model_used"],
                    feedback_type=feedback_data["feedback_type"],
                    user_comment=feedback_data["user_comment"],
                    metadata=feedback_data["metadata"]
                )
            
            # Also store in original feedback store
            feedback_store.save_feedback(
                question=response["question"],
                answer=response["answer"],
                rating=5 if thumbs_up else 1,  # Convert to 1-5 scale
                comment=feedback_comment,
                question_type=response["question_type"],
                passage_id=response["passage_info"]["id"],
                generation_method=response["generation_method"],
                session_id=session.session_id if session else None,
                metadata=feedback_data["metadata"]
            )
            
            # Mark feedback as submitted
            st.session_state.feedback_submitted = True
            
            # Show success message
            st.success("Thank you for your feedback!")
        
        # Follow-up question section
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<strong>Ask a follow-up question:</strong>", unsafe_allow_html=True)
        
        follow_up_col1, follow_up_col2 = st.columns([3, 1])
        
        with follow_up_col1:
            follow_up = st.text_input(
                "Follow-up question",
                placeholder="e.g., Can you explain that in simpler terms?",
                key="follow_up",
                label_visibility="collapsed"
            )
        
        with follow_up_col2:
            follow_up_button = st.button("Ask Follow-up", key="ask_followup")
        
        if follow_up_button and follow_up:
            # If feedback has been submitted, update the follow-up question
            if st.session_state.feedback_submitted and ENABLE_FEEDBACK_LOGGING:
                # Get the most recent feedback entry
                from feedback_logger import get_feedback_entries
                feedback_entries = get_feedback_entries(limit=1)
                if feedback_entries:
                    last_entry = feedback_entries[0]
                    # Log a new entry with the follow-up included
                    log_feedback(
                        question=last_entry["question"],
                        answer=last_entry["answer"],
                        passage=last_entry["passage"],
                        model_used=last_entry["model_used"],
                        feedback_type=last_entry["feedback_type"],
                        user_comment=last_entry["user_comment"],
                        follow_up_question=follow_up,
                        metadata=feedback_data["metadata"] if "feedback_data" in locals() else None
                    )
            
            # Process the follow-up question
            with st.spinner("Processing follow-up..."):
                follow_up_response = process_question(
                    question=follow_up,
                    retriever=retriever,
                    answer_generator=answer_generator,
                    session=session,
                    use_ollama=use_ollama,
                    ollama_model=ollama_model,
                    use_transformers=use_transformers,
                    use_reranker=use_reranker,
                    top_n=top_n,
                    is_follow_up=True,
                    debug=ENABLE_DEBUG_MODE and show_debug
                )
                
                # Clean the follow-up response before storing
                follow_up_response = clean_answer_text(follow_up_response)
                
                # Store response in session state
                st.session_state.last_response = follow_up_response
            
            # Display the follow-up answer
            display_followup_card(follow_up_response)
            
            # Refresh the page to update conversation history
            st.experimental_rerun()
        
        # Debug information if enabled
        if ENABLE_DEBUG_MODE and show_debug:
            with st.expander("Debug Information", expanded=False):
                # Performance metrics
                st.markdown("### Performance")
                st.metric("Execution Time", f"{response['execution_time']:.2f} seconds")
                
                # Session info
                st.markdown("### Session Info")
                st.json({
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "interaction_count": len(session.history)
                })
                
                # Classification info
                st.markdown("### Question Classification")
                st.json(response["classification"])
                
                # Reranking explanation
                if response["ranking_explanation"]:
                    st.markdown("### Reranking Explanation")
                    st.json(response["ranking_explanation"])
                
                # Follow-up information if applicable
                if response["is_follow_up"]:
                    st.markdown("### Follow-up Context")
                    st.text(response["debug"]["conversation_context"])
                    st.markdown("#### Augmented Query")
                    st.text(response["debug"]["augmented_query"])
                
                # Answer generation details
                st.markdown("### Answer Generation")
                generation_info = {
                    "method": response['generation_method'],
                    "confidence": response['answer_confidence'],
                    "model": ollama_model if use_ollama else ("gpt2" if use_transformers else "rule-based")
                }
                st.json(generation_info)
    
    # If no response yet, show sample questions
    if "last_response" not in st.session_state or not st.session_state.last_response:
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; margin-bottom: 1rem;'><strong>Sample questions to try:</strong></div>", unsafe_allow_html=True)
        
        # Display sample questions as clickable buttons
        samples_col1, samples_col2 = st.columns(2)
        
        with samples_col1:
            st.markdown("""
            - What is the bourgeoisie?
            - How does Marx define communism?
            - What is class struggle?
            - Why does Marx criticize capitalism?
            """)
        
        with samples_col2:
            st.markdown("""
            - When was the Communist Manifesto written?
            - What examples of worker exploitation does Marx provide?
            - How does Marx compare capitalism to feudalism?
            - What causes revolution according to Marx?
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Made with ❤️ for philosophical exploration | Powered by Ollama</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 