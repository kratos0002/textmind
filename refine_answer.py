#!/usr/bin/env python3
"""
refine_answer.py - Utility for refining answers based on follow-up questions

This module provides functionality to refine a previously generated answer
using a follow-up question, leveraging Ollama LLMs when available and
falling back to rule-based generation when needed.
"""

import os
import sys
import logging
import re
from typing import Dict, Any, Optional, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the scripts directory to path to allow importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
sys.path.append(scripts_dir)

# Import required modules
try:
    # Try direct import first
    from scripts.answer_generator import (
        AnswerGenerator, 
        OLLAMA_AVAILABLE, 
        generate_answer_with_ollama,
        strip_html_tags  # Import the HTML stripping function
    )
    
    # Define a wrapper function for rule-based generation
    def generate_answer_rule_based(question: str, passage: Dict[str, Any], question_type: str) -> Dict[str, Any]:
        # Create an AnswerGenerator instance
        generator = AnswerGenerator(use_llm=False, use_ollama=False)
        return generator.generate_answer(question, passage, question_type, method="rule")
        
except ImportError:
    # If direct import fails, try alternative import path
    try:
        sys.path.append(os.path.dirname(current_dir))  # Add parent directory 
        from answer_generator import (
            AnswerGenerator, 
            OLLAMA_AVAILABLE,
            strip_html_tags  # Import the HTML stripping function
        )
        
        # Define wrapper functions for answer generation methods
        def generate_answer_with_ollama(prompt: str, model: str = "mistral") -> Dict[str, Any]:
            generator = AnswerGenerator(use_llm=False, use_ollama=True, ollama_model=model)
            # Ollama doesn't use passages directly in this use case, so create a dummy
            dummy_passage = {"text": prompt, "paragraph_id": "refinement"}
            return generator.generate_answer(
                question=prompt,
                passage=dummy_passage,
                question_type="Context",
                method="ollama"
            )
        
        def generate_answer_rule_based(question: str, passage: Dict[str, Any], question_type: str) -> Dict[str, Any]:
            generator = AnswerGenerator(use_llm=False, use_ollama=False)
            return generator.generate_answer(question, passage, question_type, method="rule")
            
    except ImportError:
        logger.error("Failed to import AnswerGenerator. Make sure answer_generator.py exists.")
        # Define fallback values and functions
        OLLAMA_AVAILABLE = False
        
        # Define our own strip_html_tags function if import fails
        def strip_html_tags(text):
            """Remove HTML tags from text"""
            if not text:
                return ""
            return re.sub(r'<[^>]*>', '', text)
        
        def generate_answer_rule_based(question: str, passage: Dict[str, Any], question_type: str) -> Dict[str, Any]:
            return {"answer": f"I cannot refine the answer without the proper modules.", "confidence": 0.0, "method": "fallback"}
        
        def generate_answer_with_ollama(prompt: str, model: str = "mistral") -> Dict[str, Any]:
            return {"answer": f"Ollama is not available for refining the answer.", "confidence": 0.0, "method": "fallback"}


def refine_answer(
    original_question: str,
    original_answer: str,
    follow_up_question: str,
    model: str = "ollama",
    ollama_model: str = "mistral"
) -> str:
    """
    Refines a previously generated answer based on a follow-up question.
    
    Args:
        original_question: The initial question that was asked
        original_answer: The answer that was given to the original question
        follow_up_question: The follow-up question asking for refinement
        model: The model to use for refinement ('ollama' or 'rule')
        ollama_model: The specific Ollama model to use if model='ollama'
        
    Returns:
        A refined answer as a string
    """
    logger.info(f"Refining answer for follow-up: {follow_up_question}")
    
    # Construct the prompt for refinement
    prompt = f"""You previously answered the question: "{original_question}" with the answer: "{original_answer}".
The user now asks a follow-up question: "{follow_up_question}".
Please refine and expand your original answer to better address the follow-up."""
    
    refined_answer = ""
    
    # Try the preferred model first
    if model.lower() == "ollama" and OLLAMA_AVAILABLE:
        try:
            logger.info(f"Attempting to refine using Ollama with model {ollama_model}")
            result = generate_answer_with_ollama(prompt, model=ollama_model)
            refined_answer = result.get("answer", "")
            logger.info("Successfully refined answer using Ollama")
        except Exception as e:
            logger.warning(f"Error using Ollama for refinement: {str(e)}")
            logger.info("Falling back to rule-based refinement")
            model = "rule"  # Fall back to rule-based
    
    # If Ollama failed or wasn't requested, use rule-based approach
    if model.lower() == "rule" or not refined_answer:
        try:
            logger.info("Using rule-based approach for refinement")
            refined_answer = rule_based_refinement(original_question, original_answer, follow_up_question)
        except Exception as e:
            logger.error(f"Rule-based refinement failed: {str(e)}")
            # Ultimate fallback
            refined_answer = f"I apologize, but I couldn't refine my answer to address '{follow_up_question}'. My original answer was: {original_answer}"
    
    # Ensure any HTML tags are stripped
    return strip_html_tags(refined_answer)


def rule_based_refinement(original_question: str, original_answer: str, follow_up_question: str) -> str:
    """
    Rule-based approach to refine an answer based on common follow-up patterns.
    
    Args:
        original_question: The initial question that was asked
        original_answer: The answer that was given to the original question
        follow_up_question: The follow-up question asking for refinement
        
    Returns:
        A refined answer as a string
    """
    # Convert to lowercase for easier pattern matching
    follow_up_lower = follow_up_question.lower()
    
    # Check for different types of follow-up requests
    if any(phrase in follow_up_lower for phrase in ["explain", "elaborate", "tell me more", "what do you mean"]):
        return f"{original_answer}\n\nTo elaborate further: This concept is a key part of Marx's analysis of capitalism and class struggle. In the Communist Manifesto, Marx explains that these economic and social relations are not static but evolve through historical development."
    
    elif any(phrase in follow_up_lower for phrase in ["simpler", "simple terms", "easier to understand", "eli5"]):
        return f"In simpler terms: {original_answer.split('. ')[0]}. This is basically about how the economic system works according to Marx's theory."
    
    elif any(phrase in follow_up_lower for phrase in ["example", "instance", "illustration"]):
        return f"{original_answer}\n\nFor example, in the Communist Manifesto, Marx cites the industrial revolution as a period when these dynamics were clearly visible in society."
    
    elif any(phrase in follow_up_lower for phrase in ["compare", "contrast", "difference", "similar"]):
        return f"{original_answer}\n\nIn contrast to other theories of his time, Marx's view was unique in focusing on the material and economic basis of social relations rather than on abstract ideas."
    
    elif any(phrase in follow_up_lower for phrase in ["why", "reason", "purpose", "cause"]):
        return f"{original_answer}\n\nThe reason this is significant in Marx's theory is that it forms the foundation for understanding how economic systems evolve and ultimately why he believed capitalism would eventually be replaced by communism."
    
    # Default enhancement if no specific pattern is matched
    return f"{original_answer}\n\nRegarding your follow-up about '{follow_up_question}': The Communist Manifesto provides additional context on this topic, emphasizing the historical importance of class relations in shaping society."


if __name__ == "__main__":
    # Test the refine_answer function
    original_question = "What is class struggle according to Marx?"
    original_answer = "According to Marx, class struggle is the conflict between the bourgeoisie (capitalist class) and the proletariat (working class). This struggle emerges from the exploitation inherent in the capitalist mode of production, where the bourgeoisie owns the means of production and extracts surplus value from the labor of the proletariat."
    
    # Test with different follow-up questions
    follow_ups = [
        "Can you explain that in simpler terms?",
        "Give me an example from the Communist Manifesto",
        "Why did Marx think this was important?",
        "How does this compare to modern views of social conflict?"
    ]
    
    print("Testing refine_answer function with various follow-up questions:\n")
    
    for i, follow_up in enumerate(follow_ups, 1):
        print(f"Test {i}: Follow-up: '{follow_up}'")
        
        # Test with Ollama if available
        if OLLAMA_AVAILABLE:
            print("\nRefinement using Ollama:")
            refined = refine_answer(original_question, original_answer, follow_up, model="ollama")
            print(refined)
        
        # Always test rule-based for comparison
        print("\nRefinement using rule-based approach:")
        refined = refine_answer(original_question, original_answer, follow_up, model="rule")
        print(refined)
        
        print("\n" + "-"*80 + "\n") 