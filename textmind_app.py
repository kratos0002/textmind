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

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="TextMind - Ask Marx Anything",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

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
from utils.concept_map import render_concept_map

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
def load_answer_generator(_retriever, use_ollama=True, ollama_model="mistral", use_transformers=False):
    """
    Load the answer generator with caching
    
    Args:
        _retriever: TextRetriever instance (renamed to avoid hash issues with streamlit)
        use_ollama: Whether to use Ollama model
        ollama_model: Which Ollama model to use
        use_transformers: Whether to use Transformers model
        
    Returns:
        AnswerGenerator instance
    """
    try:
        logger.info(f"Initializing AnswerGenerator with model settings: use_ollama={use_ollama}, ollama_model={ollama_model}, use_transformers={use_transformers}")
        
        # Create answer generator
        generator = AnswerGenerator(
            use_llm=use_transformers,  # AnswerGenerator uses use_llm instead of use_transformers
            model_name="gpt2",
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            retriever=_retriever  # Pass the retriever to AnswerGenerator
        )
        
        return generator
    except Exception as e:
        logger.error(f"Error initializing answer generator: {str(e)}")
        # Return a minimal generator that will display errors
        return AnswerGenerator(use_llm=False, use_ollama=False, retriever=_retriever)

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
            # Get the previous passage
            previous_passage = previous_interaction.get('metadata', {}).get('passage_text', '')
            
            # Use refine_answer to get a refined answer based on the previous interaction
            refined_answer = refine_answer(
                original_question=previous_interaction.get('question', ''),
                original_answer=previous_interaction.get('answer', ''),
                follow_up_question=question,
                model="ollama" if use_ollama else "rule",
                ollama_model=ollama_model,
                passage_text=previous_passage
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
    
    # Handle unknown question types with a generic intro message
    unknown_intro = ""
    if question_type == "Unknown":
        from scripts.question_classifier import handle_unknown_question_type
        unknown_intro = handle_unknown_question_type(question) + " "
    
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
        answer = "I couldn't find relevant information to answer your question."
        answer_confidence = 0.0
        generation_method = "default"
    else:
        # For Unknown question types, prefer ollama if available or fall back to rule-based
        if question_type == "Unknown" and use_ollama:
            generation_method = "ollama"
        else:
            # Default method selection
            generation_method = "ollama" if use_ollama else "rule"
            if use_transformers:
                generation_method = "llm"  # Changed from "transformers" to "llm" to match AnswerGenerator's method naming
        
        # Debug log for document_chunks
        if debug:
            if 'document_chunks' in top_passage:
                print(f"DEBUG: Document chunks available in top_passage with {len(top_passage['document_chunks'])} items")
            else:
                print("DEBUG: No document_chunks available in top_passage")
        
        # Return the answer with metadata
        answer_data = answer_generator.generate_answer(
            question=question,
            passage=top_passage,
            question_type=question_type,
            method=generation_method,
            include_passage_info=True
        )
        
        # Debug log for source_location
        if debug:
            print(f"DEBUG: Source location found: {answer_data.get('source_location', {}).get('found', False)}")
            if 'source_location' in answer_data and answer_data['source_location'].get('found', False):
                print(f"DEBUG: Source location section: {answer_data['source_location'].get('section', 'None')}")
                print(f"DEBUG: Source location paragraph: {answer_data['source_location'].get('paragraph_index', 'None')}")
                print(f"DEBUG: Source location concepts: {answer_data['source_location'].get('concepts', [])}")
        
        # Prepend the generic intro for unknown question types
        if unknown_intro and answer_data.get("answer"):
            answer_data["answer"] = unknown_intro + answer_data["answer"]
        
        # Extract the answer and confidence
        answer = answer_data.get("answer", "No answer generated.")
        answer_confidence = answer_data.get("confidence", 0.0)
        generation_method = answer_data.get("method", generation_method)
        source_location = answer_data.get("source_location", {"found": False})
    
    # Clean the answer (remove any HTML tags)
    answer = strip_html_tags(answer)
    
    # Step 6: Create the response
    response = {
        "question": question,
        "question_type": question_type,
        "confidence": confidence,
        "answer": answer,
        "answer_confidence": answer_confidence,
        "generation_method": generation_method,
        "execution_time": (datetime.datetime.now() - start_time).total_seconds(),
        "is_follow_up": is_follow_up,
        "session_id": session.session_id if session else None,
        "passage": top_passage,
        "passage_info": {
            "id": top_passage.get("id", "") if top_passage else "",
            "text": top_passage.get("text", "") if top_passage else "",
            "hybrid_score": top_passage.get("hybrid_score", 0.0) if top_passage else 0.0,
            "semantic_score": top_passage.get("score", 0.0) if top_passage else 0.0,
            "keyword_score": top_passage.get("keyword_score", 0.0) if top_passage else 0.0,
            "matched_keywords": top_passage.get("matched_keywords", []) if top_passage else []
        },
        "source_location": source_location,
        "all_passages": reranked_passages,
        "ranking_explanation": ranking_explanation,
        "classification": classification,
        "debug": {
            "conversation_context": conversation_context,
            "is_unknown_type": question_type == "Unknown",
            "unknown_intro_used": bool(unknown_intro)
        }
    }
    
    # Step 7: Add to session history
    if session:
        metadata = {
            "question_type": question_type,
            "method": generation_method,
            "passage_id": top_passage.get("id", "") if top_passage else "",
            "passage_text": top_passage.get("text", "") if top_passage else "",
            "is_follow_up": is_follow_up,
            "execution_time": response["execution_time"]
        }
        session.add_interaction(
            question=question,
            answer=answer,
            question_type=question_type,
            metadata=metadata
        )
    
    return response

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
    Display the answer card with question, answer, type, and feedback
    """
    if not response:
        return
        
    # Extract fields from response
    question = response.get('question', '')
    question_type = response.get('question_type', 'Unknown')
    answer = response.get('answer', '')
    confidence = response.get('confidence', 0)
    generation_method = response.get('generation_method', '')
    execution_time = response.get('execution_time', 0)
    passage_id = response.get('passage_info', {}).get('id', '')
    passage_text = response.get('passage_info', {}).get('text', '')
    
    # Extract source location information if available
    source_location = {}
    if 'source_location' in response:
        source_location = response.get('source_location', {})
    else:
        # Check if it's nested under passage_info
        source_location = response.get('passage_info', {}).get('source_location', {})
    
    # Format the card
    st.markdown(f"### {question}")
    
    # Show answer with HTML cleanly stripped
    st.markdown(answer)
    
    # Add a faint hint about the source with enhanced location information
    if passage_text:
        source_hint = f"*Source information from passage {passage_id}*"
        
        # Add location details if found
        if source_location and source_location.get('found', False):
            section = source_location.get('section', '')
            paragraph_index = source_location.get('paragraph_index')
            concepts = source_location.get('concepts', [])
            
            location_details = []
            if section:
                location_details.append(f"Section: {section}")
            if paragraph_index is not None:
                location_details.append(f"Paragraph: {paragraph_index}")
            
            if location_details:
                source_hint += f" ({', '.join(location_details)})"
            
            # Add concepts if available
            if concepts:
                concept_list = ', '.join([f"#{c}" for c in concepts[:3]])  # Show up to 3 concepts
                source_hint += f" | Concepts: {concept_list}"
        
        st.markdown(f"<div style='color: #666; font-size: 0.8em; margin-top: 0.5em; margin-bottom: 1em;'>{source_hint}</div>", unsafe_allow_html=True)
    
    # Format metadata as clean horizontal icons/badges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if question_type and question_type != "Unknown":
            user_friendly_type = question_type
            st.markdown(f"📚 {user_friendly_type}")
        
    with col2:
        if generation_method == "ollama":
            method_label = "AI-Generated" 
        elif generation_method == "transformers":
            method_label = "ML-Generated"
        elif generation_method == "rule":
            method_label = "Rule-Based"
        elif generation_method == "refinement":
            method_label = "Refined Answer"
        else:
            method_label = "Generated"
            
        st.markdown(f"🤖 {method_label}")
        
    with col3:
        if execution_time:
            st.markdown(f"⏱️ {execution_time:.2f}s")
    
    # Add thumbs up/down feedback
    col_feedback1, col_feedback2, _ = st.columns([1, 1, 4])
    
    with col_feedback1:
        st.button("👍", key=f"thumbs_up_{question}", help="This answer was helpful")
        
    with col_feedback2:
        st.button("👎", key=f"thumbs_down_{question}", help="This answer wasn't helpful")

def display_followup_card(response):
    """
    Display a card for follow-up questions with original question-answer context
    """
    if not response or not response.get('debug', {}).get('refinement'):
        return
        
    original_question = response.get('debug', {}).get('original_question', '')
    original_answer = response.get('debug', {}).get('original_answer', '')
    
    if original_question and original_answer:
        with st.expander("View original question and answer", expanded=False):
            st.markdown(f"**Original Question:** {original_question}")
            st.markdown(f"**Original Answer:** {original_answer}")
            
            # Add source hint if available
            passage_text = response.get('passage_info', {}).get('text', '')
            passage_id = response.get('passage_info', {}).get('id', '')
            source_location = response.get('source_location', {})
            
            if passage_text:
                source_hint = f"*Source information from passage {passage_id}*"
                
                # Add location details if found
                if source_location and source_location.get('found', False):
                    section = source_location.get('section', '')
                    paragraph_index = source_location.get('paragraph_index')
                    concepts = source_location.get('concepts', [])
                    
                    location_details = []
                    if section:
                        location_details.append(f"Section: {section}")
                    if paragraph_index is not None:
                        location_details.append(f"Paragraph: {paragraph_index}")
                    
                    if location_details:
                        source_hint += f" ({', '.join(location_details)})"
                    
                    # Add concepts if available
                    if concepts:
                        concept_list = ', '.join([f"#{c}" for c in concepts[:3]])  # Show up to 3 concepts
                        source_hint += f" | Concepts: {concept_list}"
                
                st.markdown(f"<div style='color: #666; font-size: 0.8em; margin-top: 0.5em;'>{source_hint}</div>", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Tabs for different features
    tab_qa, tab_concept_map = st.tabs(["Ask Questions", "Concept Map"])
    
    with tab_qa:
        # Main Q&A interface
        display_qa_interface()
    
    with tab_concept_map:
        # Concept map visualization
        render_concept_map()

def display_qa_interface():
    """Display the main Q&A interface"""
    # Load the app header and style
    st.title("TextMind: Ask Marx Anything")
    st.markdown("""
    <div class='header-description'>
        Ask questions about <em>The Communist Manifesto</em> by Karl Marx and Friedrich Engels
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar settings
    st.sidebar.title("Settings")
    
    # Model selection in sidebar
    use_ollama = False  # Default
    use_transformers = False
    ollama_model = DEFAULT_OLLAMA_MODEL
    
    model_option = st.sidebar.radio(
        "Model for answer generation",
        options=["Rule-based", "Ollama", "Transformers"],
        index=0,
        help="Select the model to use for generating answers"
    )
    
    if model_option == "Ollama":
        use_ollama = True
        ollama_model = st.sidebar.selectbox(
            "Ollama model",
            options=["mistral", "llama3", "llama2", "vicuna"],
            index=0,
            help="Select which Ollama model to use"
        )
    elif model_option == "Transformers":
        use_transformers = True
    
    # Additional settings
    use_reranker = st.sidebar.checkbox("Use passage reranking", value=True)
    top_n = st.sidebar.slider("Number of passages to retrieve", min_value=1, max_value=10, value=MAX_PASSAGES)
    
    # Load retriever and answer generator
    retriever = load_retriever(DATA_FILE, EMBEDDINGS_DIR)
    answer_generator = load_answer_generator(
        _retriever=retriever,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        use_transformers=use_transformers
    )
    
    # Create or load session if enabled
    session = None
    if ENABLE_SESSION_MANAGEMENT:
        # Get or create session ID
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        session_id = st.session_state.session_id
        session = session_manager.get_session(session_id)
        
        if session:
            logger.info(f"Loaded existing session: {session_id}")
        else:
            session = session_manager.create_session()
            st.session_state.session_id = session.session_id
            logger.info(f"Created new session: {session_id}")
    
    # Initialize state for feedback
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    
    # Question input
    st.markdown("<div class='question-container'>", unsafe_allow_html=True)
    
    # Radio button for question type
    is_follow_up = st.radio(
        "Question type",
        options=["New Question", "Follow-up Question"],
        index=0,
        horizontal=True,
        help="Select if this is a new question or a follow-up to your previous question"
    ) == "Follow-up Question"
    
    # Create a form for the question
    with st.form(key="question_form"):
        user_question = st.text_input(
            "Ask a question about Marx's Communist Manifesto",
            placeholder="e.g., What is class struggle according to Marx?",
            key="question"
        )
        
        ask_button = st.form_submit_button("Ask Question")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sample questions expander
    with st.expander("Sample questions to try", expanded=False):
        st.markdown("""
        - **Definition**: What is the bourgeoisie?
        - **Explanation**: How does Marx define communism?
        - **Historical**: When was the Communist Manifesto written?
        - **Analysis**: Why does Marx believe capitalism will collapse?
        - **Evidence**: What examples does Marx provide for worker exploitation?
        - **Comparison**: How does Marx compare capitalism to feudalism?
        - **Concept**: What is class struggle?
        - **Reasoning**: Why does Marx criticize the bourgeoisie?
        - **Cause-Effect**: What causes revolution according to Marx?
        """)
    
    # Debug mode toggle in expanded section
    with st.sidebar.expander("Advanced Options", expanded=False):
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
                # Get the original response
                original_response = st.session_state.last_response
                original_question = original_response.get("question", "")
                original_answer = original_response.get("answer", "")
                original_passage = original_response.get("passage", {})
                passage_text = original_response.get("passage_info", {}).get("text", "")
                document_chunks = original_response.get("all_passages", [])
                
                # Check if we should use the refinement approach
                if original_question and original_answer and passage_text:
                    from refine_answer import refine_answer
                    
                    # Use the refine_answer function with source location support
                    refined = refine_answer(
                        original_question=original_question,
                        original_answer=original_answer,
                        follow_up_question=follow_up,
                        model="ollama" if use_ollama else "rule",
                        ollama_model=ollama_model,
                        passage_text=passage_text,
                        document_chunks=document_chunks,
                        original_passage=original_passage
                    )
                    
                    # Create a response with the refined answer
                    follow_up_response = {
                        "question": follow_up,
                        "question_type": original_response.get("question_type", "Unknown"),
                        "confidence": original_response.get("confidence", 0.0),
                        "answer": refined.get("answer", ""),
                        "answer_confidence": refined.get("confidence", 0.7),
                        "generation_method": refined.get("method", "refinement"),
                        "execution_time": 0.0,  # We're not measuring this separately
                        "is_follow_up": True,
                        "session_id": session.session_id if session else None,
                        "passage": original_passage,
                        "passage_info": original_response.get("passage_info", {}),
                        "source_location": refined.get("source_location", {"found": False}),
                        "all_passages": document_chunks,
                        "debug": {
                            "refinement": True,
                            "original_question": original_question,
                            "original_answer": original_answer,
                            "conversation_context": f"Q: {original_question}\nA: {original_answer}\nFollow-up: {follow_up}"
                        }
                    }
                    
                    # Add to session history
                    if session:
                        metadata = {
                            "question_type": original_response.get("question_type", "Unknown"),
                            "method": refined.get("method", "refinement"),
                            "passage_id": original_response.get("passage_info", {}).get("id", ""),
                            "passage_text": passage_text,
                            "is_follow_up": True,
                            "refinement": True,
                            "original_question": original_question,
                            "execution_time": 0.0
                        }
                        session.add_interaction(
                            question=follow_up,
                            answer=refined.get("answer", ""),
                            question_type=original_response.get("question_type", "Unknown"),
                            metadata=metadata
                        )
                    
                    # Store the response and display it
                    st.session_state.last_response = follow_up_response
                    
                    # Display the follow-up response
                    st.markdown("<hr>", unsafe_allow_html=True)
                    display_card(follow_up_response)
                else:
                    # Fallback to regular question processing
                    with st.spinner("Processing as new question..."):
                        fallback_response = process_question(
                            question=follow_up,
                            retriever=retriever,
                            answer_generator=answer_generator,
                            session=session,
                            use_ollama=use_ollama,
                            ollama_model=ollama_model,
                            use_transformers=use_transformers,
                            use_reranker=use_reranker,
                            top_n=top_n,
                            is_follow_up=True,  # Mark as follow-up for context
                            debug=ENABLE_DEBUG_MODE and show_debug
                        )
                        
                        # Clean and store the response
                        fallback_response = clean_answer_text(fallback_response)
                        st.session_state.last_response = fallback_response
                        
                        # Display the follow-up response
                        st.markdown("<hr>", unsafe_allow_html=True)
                        display_card(fallback_response)

# Run the main function
if __name__ == "__main__":
    main() 