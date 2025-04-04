#!/usr/bin/env python3
"""
Answer Generator for TextMind Q&A System.

This module takes reranked passages from the retriever and generates
concise, contextually accurate answers based on the passage content
and the question type.
"""

import re
import string
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import os
import sys
import subprocess
import shlex
import requests
from pathlib import Path

# Add this function to strip HTML tags
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
    
    # Use a very aggressive regex pattern to remove anything that looks like HTML
    clean_text = re.sub(r'<[^>]*>', '', text)
    
    # Also fix common escaped HTML tags that might appear
    clean_text = clean_text.replace("&lt;", "<").replace("&gt;", ">")
    clean_text = re.sub(r'<[^>]*>', '', clean_text)
    
    return clean_text

# Add parent directory to path for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Try to import transformers for LLM-based generation
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers library not available. Install with 'pip install transformers'")
    TRANSFORMERS_AVAILABLE = False

# Check if Ollama is available
OLLAMA_AVAILABLE = False
try:
    result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
    OLLAMA_AVAILABLE = result.returncode == 0
    if OLLAMA_AVAILABLE:
        print("Ollama is available for local LLM generation")
except Exception:
    print("Ollama check failed, assuming it's not available")

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Answer generator for TextMind that combines rule-based templates with LLM generation."""
    
    def __init__(self, use_llm: bool = True, model_name: str = "gpt2", 
                 use_ollama: bool = False, ollama_model: str = "mistral"):
        """
        Initialize the answer generator
        
        Args:
            use_llm: Whether to use LLM for answer generation
            model_name: Name of the model to use for LLM generation
            use_ollama: Whether to use Ollama for local LLM generation
            ollama_model: Name of the Ollama model to use
        """
        self.use_llm = use_llm and (TRANSFORMERS_AVAILABLE or (use_ollama and OLLAMA_AVAILABLE))
        self.model_name = model_name
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.ollama_model = ollama_model
        self.llm = None
        
        # Initialize LLM if requested and not using Ollama
        if self.use_llm and not self.use_ollama and TRANSFORMERS_AVAILABLE:
            try:
                self._init_llm()
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers LLM: {str(e)}")
                self.use_llm = self.use_ollama and OLLAMA_AVAILABLE  # Fall back to Ollama if available
                
        # Log configuration
        if self.use_ollama:
            logger.info(f"Using Ollama with model {self.ollama_model}")
        elif self.use_llm:
            logger.info(f"Using Transformers with model {self.model_name}")
        else:
            logger.info("Using rule-based answer generation only")
    
    def _init_llm(self):
        """Initialize the LLM for generation"""
        try:
            logger.info(f"Initializing LLM with model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            self.llm = None
            raise
    
    def generate_answer(self, 
                        question: str, 
                        passage: Dict[str, Any], 
                        question_type: str = None,
                        method: str = "auto") -> Dict[str, Any]:
        """
        Generate an answer using the top passage
        
        Args:
            question: The user's question
            passage: Dictionary containing the top passage and metadata
            question_type: Type of question (Definition, Argument, etc.)
            method: Generation method - 'rule', 'llm', 'ollama', or 'auto'
            
        Returns:
            Dictionary with generated answer and metadata
        """
        if not passage or "text" not in passage:
            return {
                "answer": "I couldn't find a relevant answer to your question.",
                "confidence": 0.0,
                "method": "fallback",
                "passage_id": None
            }
        
        passage_text = passage["text"]
        
        # Determine the method to use
        use_method = method.lower()
        if use_method == "auto":
            if self.use_ollama:
                use_method = "ollama"
            elif self.use_llm and self.llm is not None:
                use_method = "llm"
            else:
                use_method = "rule"
        
        # Generate answer using selected method with fallbacks
        answer = ""
        confidence = 0.0
        method_used = "fallback"
        
        try:
            # Try Ollama generation
            if use_method == "ollama":
                if self.use_ollama and OLLAMA_AVAILABLE:
                    ollama_answer, ollama_confidence = self.generate_answer_ollama(
                        question, passage_text, question_type
                    )
                    answer = ollama_answer
                    confidence = ollama_confidence
                    method_used = "ollama"
                else:
                    logger.warning("Ollama requested but not available, falling back")
                    use_method = "llm" if self.use_llm and self.llm is not None else "rule"
            
            # Try LLM generation
            if use_method == "llm":
                if self.use_llm and self.llm is not None:
                    llm_answer, llm_confidence = self._llm_based_generation(
                        question, passage_text, question_type
                    )
                    answer = llm_answer
                    confidence = llm_confidence
                    method_used = "llm"
                else:
                    logger.warning("LLM requested but not available, falling back to rule-based")
                    use_method = "rule"
            
            # Use rule-based as final option
            if use_method == "rule" or not answer:
                rule_answer, rule_confidence = self._rule_based_generation(
                    question, passage_text, question_type
                )
                answer = rule_answer
                confidence = rule_confidence
                method_used = "rule"
                
        except Exception as e:
            logger.error(f"Error generating answer with {use_method}: {str(e)}")
            # Final fallback to rule-based
            rule_answer, rule_confidence = self._rule_based_generation(
                question, passage_text, question_type
            )
            answer = rule_answer
            confidence = rule_confidence
            method_used = "rule"
        
        # Return the answer with metadata
        answer_data = {
            "answer": strip_html_tags(answer),
            "confidence": confidence,
            "method": method_used,
            "passage_id": passage.get("paragraph_id", None),
            "passage_text": passage_text,
            "matched_keywords": passage.get("matched_keywords", []),
            "keyword_score": passage.get("keyword_score", 0.0),
            "semantic_score": passage.get("score", 0.0),
            "hybrid_score": passage.get("hybrid_score", 0.0)
        }
        
        return answer_data

    def generate_answer_ollama(self,
                              question: str,
                              passage_text: str,
                              question_type: str = None) -> Tuple[str, float]:
        """
        Generate answer using Ollama
        
        Args:
            question: The user's question
            passage_text: Text of the top passage
            question_type: Type of question
            
        Returns:
            Tuple of (answer, confidence)
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama is not available")
            return "Ollama is not available for generation.", 0.0
            
        # Clean passage text
        cleaned_passage = self._clean_passage(passage_text)
        
        # Determine question type if not provided
        if not question_type:
            question_type = self._infer_question_type(question)
            
        # Create prompt for Ollama
        prompt = self._create_ollama_prompt(question, cleaned_passage, question_type)
        
        try:
            # Two options for calling Ollama: subprocess or API
            # First try the API approach
            try:
                answer = self._call_ollama_api(prompt)
            except Exception as e:
                logger.warning(f"Ollama API call failed: {str(e)}, falling back to subprocess")
                answer = self._call_ollama_subprocess(prompt)
            
            # Apply post-processing to clean up the answer
            answer = self._post_process_ollama_answer(answer)
            
            # Estimate confidence based on answer quality
            confidence = self._estimate_ollama_confidence(answer, question, cleaned_passage)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error in Ollama generation: {str(e)}")
            return f"Failed to generate Ollama-based answer: {str(e)}", 0.0
    
    def _call_ollama_api(self, prompt: str) -> str:
        """
        Call Ollama using the HTTP API
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Generated text from Ollama
        """
        # Try to connect to Ollama API (usually on localhost:11434)
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "stream": False
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    def _call_ollama_subprocess(self, prompt: str) -> str:
        """
        Call Ollama using subprocess
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Generated text from Ollama
        """
        try:
            # Run Ollama with the specified model
            result = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt.encode(),
                capture_output=True,
                check=True
            )
            # Extract the generated text
            output = result.stdout.decode().strip()
            
            # Remove the prompt from the output to get just the generated answer
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
                
            return output
        except subprocess.CalledProcessError as e:
            logger.error(f"Error calling Ollama subprocess: {str(e)}")
            raise
    
    def _create_ollama_prompt(self, 
                             question: str, 
                             passage: str, 
                             question_type: str = None) -> str:
        """
        Create prompt for Ollama-based generation
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            question_type: Type of question
            
        Returns:
            Prompt string for Ollama
        """
        # Determine question type if not provided
        if not question_type:
            question_type = self._infer_question_type(question)
            
        # Create a prompt template based on question type
        template = "You are a helpful assistant. Based on the passage below, answer the following question.\n\n"
        template += f"Question Type: {question_type}\n"
        template += f"Question: {question}\n\n"
        template += f"Context:\n{passage[:500]}\n\n"
        template += "Your Answer: "
        
        return template
    
    def _post_process_ollama_answer(self, answer: str) -> str:
        """
        Clean up the Ollama-generated answer
        
        Args:
            answer: Raw answer from Ollama
            
        Returns:
            Cleaned answer
        """
        # Remove any text after multiple newlines (likely model's continuation)
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0]
        
        # Remove common prefixes from LLM outputs
        answer = re.sub(r"^(Based on the passage:|According to the passage:|The answer is:)\s*", "", answer, flags=re.IGNORECASE)
        
        # Remove leading/trailing quotes if present
        answer = answer.strip('"\'')
        
        # Ensure the answer ends with proper punctuation
        if answer and not answer[-1] in ".!?":
            answer += "."
            
        return answer
    
    def _estimate_ollama_confidence(self, 
                                   answer: str, 
                                   question: str, 
                                   passage: str) -> float:
        """
        Estimate confidence for Ollama-generated answer
        
        Args:
            answer: Generated answer
            question: Original question
            passage: Passage text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Start with higher base confidence for Ollama models (they're generally better than tiny models)
        confidence = 0.8
        
        # Check answer length - penalize very short or very long answers
        if len(answer) < 20:
            confidence -= 0.3  # Severe penalty for very short answers
        elif len(answer) < 50:
            confidence -= 0.1  # Minor penalty for somewhat short answers
        elif len(answer) > 300:
            confidence -= 0.1  # Minor penalty for very long answers
            
        # Check if answer contains content from passage (some overlap is good)
        passage_words = set(passage.lower().split())
        answer_words = set(answer.lower().split())
        
        if answer_words:
            overlap = len(passage_words.intersection(answer_words)) / len(answer_words)
            
            # Penalize for too little or too much overlap
            if overlap < 0.3:
                confidence -= 0.2  # Significant penalty for little passage overlap
            elif overlap > 0.9:
                confidence -= 0.2  # Significant penalty for too much overlap (likely copying)
            
        # Check if answer addresses the question terms
        question_words = set(question.lower().split())
        if question_words:
            question_overlap = len(question_words.intersection(answer_words)) / len(question_words)
            
            if question_overlap < 0.2:
                confidence -= 0.2  # Penalty for not addressing question terms
        
        # Check for hallucination markers ("I don't know", "not mentioned", etc.)
        hallucination_phrases = [
            "i don't know", "not mentioned", "not provided", "not stated",
            "no information", "cannot determine", "unclear from the passage"
        ]
        
        if any(phrase in answer.lower() for phrase in hallucination_phrases):
            confidence -= 0.3  # Major penalty for hallucination markers
            
        # Ensure confidence is in range [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _rule_based_generation(self, 
                              question: str, 
                              passage_text: str, 
                              question_type: str = None) -> Tuple[str, float]:
        """
        Generate answer using rule-based templates
        
        Args:
            question: The user's question
            passage_text: Text of the top passage
            question_type: Type of question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Clean up the passage text (remove IDs, page numbers, etc.)
        cleaned_passage = self._clean_passage(passage_text)
        
        # Determine question type if not provided
        if not question_type:
            question_type = self._infer_question_type(question)
        
        # Extract key entities from the question
        entities = self._extract_entities(question)
        
        # Get the appropriate template based on question type
        if question_type.lower() == "definition":
            return self._definition_template(question, cleaned_passage, entities)
        
        elif question_type.lower() == "argument" or question_type.lower() == "why":
            return self._argument_template(question, cleaned_passage, entities)
        
        elif question_type.lower() == "context":
            return self._context_template(question, cleaned_passage, entities)
        
        elif question_type.lower() == "evidence":
            return self._evidence_template(question, cleaned_passage, entities)
        
        elif question_type.lower() == "analogy":
            return self._analogy_template(question, cleaned_passage, entities)
            
        # Default to direct passage with light formatting
        confidence = 0.6  # Medium confidence for default approach
        # Extract most relevant sentence(s)
        relevant_part = self._extract_relevant_part(question, cleaned_passage)
        return relevant_part, confidence
    
    def _definition_template(self, 
                            question: str, 
                            passage: str, 
                            entities: List[str]) -> Tuple[str, float]:
        """
        Template for definition questions
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            entities: Key entities from the question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Extract what's being defined
        term = self._extract_definition_term(question, entities)
        
        # Look for passages with pattern "X is/are/refers to..."
        definition_patterns = [
            r"(?i)" + re.escape(term) + r"\s+(?:is|are|refers to|means|defines?)\s+([^\.]+)",
            r"(?i)(?:term|concept|idea of)\s+" + re.escape(term) + r"\s+([^\.]+)",
            r"(?i)" + re.escape(term) + r"\s+(?:can be understood as|consists of|comprises)\s+([^\.]+)"
        ]
        
        for pattern in definition_patterns:
            match = re.search(pattern, passage)
            if match:
                definition = match.group(1).strip()
                answer = f"{term} is {definition}."
                return answer, 0.9  # High confidence for direct definition match
                
        # If no clear definition pattern found, identify sentences containing the term
        sentences = self._split_into_sentences(passage)
        term_sentences = [s for s in sentences if term.lower() in s.lower()]
        
        if term_sentences:
            # Use the first sentence containing the term
            best_sentence = term_sentences[0]
            
            # Format it as a definition
            if best_sentence.lower().startswith(term.lower()):
                answer = best_sentence.strip()
            else:
                # Create a definition from the sentence
                answer = f"{term} refers to {best_sentence.strip()}"
            
            return answer, 0.8
        
        # Fallback: Use the entire passage
        return f"{term} can be understood in context as: {passage[:200].strip()}", 0.5
        
    def _argument_template(self, 
                          question: str, 
                          passage: str, 
                          entities: List[str]) -> Tuple[str, float]:
        """
        Template for argument/why questions
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            entities: Key entities from the question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Look for causal language
        causal_patterns = [
            r"(?i)because\s+([^\.]+)",
            r"(?i)due to\s+([^\.]+)",
            r"(?i)as a result of\s+([^\.]+)",
            r"(?i)the reason (?:is|for this)?\s+([^\.]+)",
            r"(?i)leads to\s+([^\.]+)",
            r"(?i)caused by\s+([^\.]+)"
        ]
        
        # Check for explicit cause-effect relationships
        for pattern in causal_patterns:
            match = re.search(pattern, passage)
            if match:
                reason = match.group(1).strip()
                if any(entity.lower() in reason.lower() for entity in entities):
                    # This reason is directly related to entities in the question
                    answer = f"According to the text, this is because {reason}."
                    return answer, 0.9
        
        # If no clear causal language, extract relevant sentences with entity terms
        sentences = self._split_into_sentences(passage)
        relevant_sentences = []
        
        for sentence in sentences:
            if any(entity.lower() in sentence.lower() for entity in entities):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Combine most relevant sentences (up to 2)
            combined = " ".join(relevant_sentences[:2]).strip()
            answer = f"The text suggests that: {combined}"
            return answer, 0.7
        
        # Fallback: Use the first 1-2 sentences
        return f"Based on the passage: {'. '.join(sentences[:2])}", 0.5
        
    def _context_template(self, 
                         question: str, 
                         passage: str, 
                         entities: List[str]) -> Tuple[str, float]:
        """
        Template for context questions
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            entities: Key entities from the question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Look for temporal or location markers
        context_patterns = [
            r"(?i)in\s+(\d{4})",  # Years
            r"(?i)during the\s+([^\.]+)",  # Time periods
            r"(?i)in the\s+(context|period|era|time) of\s+([^\.]+)"  # General context
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, passage)
            if match:
                context = match.group(1).strip()
                answer = f"This occurred in the context of {context}, as the passage states."
                return answer, 0.8
        
        # Extract relevant context based on entities
        sentences = self._split_into_sentences(passage)
        entity_contexts = []
        
        for sentence in sentences:
            if any(entity.lower() in sentence.lower() for entity in entities):
                if re.search(r"(?i)during|when|where|in the|time|era|period|century|year", sentence):
                    entity_contexts.append(sentence)
        
        if entity_contexts:
            context_sent = entity_contexts[0].strip() 
            answer = f"The historical context is that {context_sent}"
            return answer, 0.7
            
        # Fallback: General context from passage
        return f"The text provides this context: {passage[:150].strip()}", 0.5
        
    def _evidence_template(self, 
                          question: str, 
                          passage: str, 
                          entities: List[str]) -> Tuple[str, float]:
        """
        Template for evidence questions
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            entities: Key entities from the question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Look for evidence markers
        evidence_patterns = [
            r"(?i)evidence\s+(?:for|of|that)\s+([^\.]+)",
            r"(?i)for example,\s+([^\.]+)",
            r"(?i)demonstrated by\s+([^\.]+)",
            r"(?i)illustrated by\s+([^\.]+)",
            r"(?i)shown by\s+([^\.]+)"
        ]
        
        for pattern in evidence_patterns:
            match = re.search(pattern, passage)
            if match:
                evidence = match.group(1).strip()
                answer = f"The text provides this evidence: {evidence}."
                return answer, 0.8
        
        # If no explicit evidence markers, look for factual statements related to entities
        sentences = self._split_into_sentences(passage)
        evidence_sentences = []
        
        for sentence in sentences:
            # Sentences containing entities and potentially factual content
            if any(entity.lower() in sentence.lower() for entity in entities):
                if re.search(r"(?i)is|are|was|were|had|have|has|show|demonstrate", sentence):
                    evidence_sentences.append(sentence)
        
        if evidence_sentences:
            # Use the most relevant evidence sentence
            evidence_sent = evidence_sentences[0].strip()
            answer = f"According to the text: {evidence_sent}"
            return answer, 0.7
            
        # Fallback: Use a relevant portion of the passage
        return f"The evidence from the passage suggests: {passage[:150].strip()}", 0.5
        
    def _analogy_template(self, 
                         question: str, 
                         passage: str, 
                         entities: List[str]) -> Tuple[str, float]:
        """
        Template for analogy/comparison questions
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            entities: Key entities from the question
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Look for comparison language
        comparison_patterns = [
            r"(?i)similar to\s+([^\.]+)",
            r"(?i)like\s+([^\.]+)",
            r"(?i)compared to\s+([^\.]+)",
            r"(?i)as with\s+([^\.]+)",
            r"(?i)parallels?\s+([^\.]+)"
        ]
        
        for pattern in comparison_patterns:
            match = re.search(pattern, passage)
            if match:
                comparison = match.group(1).strip()
                answer = f"The text draws a comparison with {comparison}."
                return answer, 0.8
                
        # If no explicit comparison found, look for sentences with comparison terms
        sentences = self._split_into_sentences(passage)
        comparison_sentences = []
        
        for sentence in sentences:
            if re.search(r"(?i)similar|like|compared|parallel|analogy|metaphor|contrast|versus", sentence):
                comparison_sentences.append(sentence)
                
        if comparison_sentences:
            comparison_sent = comparison_sentences[0].strip()
            answer = f"The text makes this comparison: {comparison_sent}"
            return answer, 0.7
            
        # Fallback: Use entity-related content
        entity_sentences = [s for s in sentences if any(entity.lower() in s.lower() for entity in entities)]
        if entity_sentences:
            return f"While no direct comparison is made, the text states: {entity_sentences[0].strip()}", 0.5
            
        # Final fallback
        return f"No direct comparison is found in the passage, which states: {passage[:150].strip()}", 0.4
    
    def _llm_based_generation(self, 
                             question: str, 
                             passage_text: str, 
                             question_type: str = None) -> Tuple[str, float]:
        """
        Generate answer using LLM
        
        Args:
            question: The user's question
            passage_text: Text of the top passage
            question_type: Type of question
            
        Returns:
            Tuple of (answer, confidence)
        """
        if not self.llm:
            return "LLM generation not available.", 0.0
            
        # Clean passage text
        cleaned_passage = self._clean_passage(passage_text)
        
        # Create prompt for LLM
        prompt = self._create_llm_prompt(question, cleaned_passage, question_type)
        
        try:
            # Generate answer with LLM
            result = self.llm(prompt, max_length=150, num_return_sequences=1, 
                             temperature=0.7, top_p=0.9)
            
            # Extract and clean the generated answer
            generated_text = result[0]["generated_text"]
            answer = generated_text.replace(prompt, "").strip()
            
            # Apply post-processing to clean up the answer
            answer = self._post_process_llm_answer(answer)
            
            # Estimate confidence based on answer quality
            confidence = self._estimate_llm_confidence(answer, question, cleaned_passage)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return f"Failed to generate LLM-based answer: {str(e)}", 0.0
    
    def _create_llm_prompt(self, 
                          question: str, 
                          passage: str, 
                          question_type: str = None) -> str:
        """
        Create prompt for LLM-based generation
        
        Args:
            question: The user's question
            passage: Cleaned passage text
            question_type: Type of question
            
        Returns:
            Prompt string for the LLM
        """
        # Determine question type if not provided
        if not question_type:
            question_type = self._infer_question_type(question)
            
        # Basic prompt structure
        prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nAnswer: "
        
        # Add type-specific instructions if available
        if question_type.lower() == "definition":
            prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nProvide a clear definition based on the passage. Answer: "
        
        elif question_type.lower() == "argument" or question_type.lower() == "why":
            prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nExplain the reasoning or argument from the passage. Answer: "
            
        elif question_type.lower() == "context":
            prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nProvide the historical or contextual information from the passage. Answer: "
            
        elif question_type.lower() == "evidence":
            prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nProvide specific evidence or examples from the passage. Answer: "
            
        elif question_type.lower() == "analogy":
            prompt = f"Passage: \"{passage[:300]}\".\n\nQuestion: {question}\n\nExplain the comparison or analogy from the passage. Answer: "
            
        return prompt
    
    def _post_process_llm_answer(self, answer: str) -> str:
        """
        Clean up the LLM-generated answer
        
        Args:
            answer: Raw answer from LLM
            
        Returns:
            Cleaned answer
        """
        # Remove any text after the first paragraph
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0]
            
        # Remove prefixes like "The answer is:" or "According to the passage:"
        answer = re.sub(r"^(The answer is:?|According to the passage:?|Based on the passage:?)\s*", "", answer)
        
        # Ensure the answer ends with proper punctuation
        if answer and not answer[-1] in ".!?":
            answer += "."
            
        return answer
    
    def _estimate_llm_confidence(self, 
                                answer: str, 
                                question: str, 
                                passage: str) -> float:
        """
        Estimate confidence for LLM-generated answer
        
        Args:
            answer: Generated answer
            question: Original question
            passage: Passage text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Start with base confidence
        confidence = 0.7
        
        # Check answer length
        if len(answer) < 10:
            confidence -= 0.2
        elif len(answer) > 200:
            confidence -= 0.1
            
        # Check if answer contains content from passage
        passage_words = set(passage.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(passage_words.intersection(answer_words)) / len(answer_words) if answer_words else 0
        
        if overlap < 0.3:
            confidence -= 0.3  # Penalize answers with little passage overlap
        elif overlap > 0.8:
            confidence -= 0.1  # Slightly penalize answers that are too close to passage (near copying)
            
        # Check if answer addresses the question
        question_words = set(question.lower().split())
        question_overlap = len(question_words.intersection(answer_words)) / len(question_words)
        
        if question_overlap < 0.2:
            confidence -= 0.2  # Penalize answers that don't address the question
            
        # Ensure confidence is in range [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _clean_passage(self, passage: str) -> str:
        """
        Clean passage text by removing artifacts and formatting
        
        Args:
            passage: Original passage text
            
        Returns:
            Cleaned passage text
        """
        # Remove page numbers and IDs
        cleaned = re.sub(r"MANIFESTO OF THE COMMUNIST PARTY \d+", "", passage)
        
        # Remove other common artifacts
        cleaned = re.sub(r"\[\d+\]", "", cleaned)  # References
        cleaned = re.sub(r"\*+", "", cleaned)  # Asterisks
        
        # Standardize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        return cleaned
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract key entities from question
        
        Args:
            question: The question text
            
        Returns:
            List of extracted entities
        """
        # Remove stop words and question words
        stop_words = {
            "what", "who", "where", "when", "why", "how", "is", "are", "was", "were",
            "do", "does", "did", "a", "an", "the", "in", "on", "at", "by", "to",
            "for", "with", "about", "against"
        }
        
        # Tokenize and filter
        words = question.lower().split()
        non_stop_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Extract potential multi-word entities
        entities = []
        i = 0
        while i < len(words):
            if words[i] not in stop_words and len(words[i]) > 2:
                # Check for 3-word entity
                if i + 2 < len(words) and words[i+1] not in stop_words and words[i+2] not in stop_words:
                    entities.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                    i += 3
                # Check for 2-word entity
                elif i + 1 < len(words) and words[i+1] not in stop_words:
                    entities.append(f"{words[i]} {words[i+1]}")
                    i += 2
                else:
                    entities.append(words[i])
                    i += 1
            else:
                i += 1
                
        # Add single words if we have few entities
        if len(entities) < 2:
            entities.extend(non_stop_words)
            
        # Deduplicate
        return list(set(entities))
    
    def _extract_definition_term(self, question: str, entities: List[str]) -> str:
        """
        Extract the term being defined from the question
        
        Args:
            question: The question text
            entities: Extracted entities
            
        Returns:
            Term being defined
        """
        # Check for "what is X" pattern
        what_is_match = re.search(r"(?i)what is(?: a| the)? ([^\?]+)", question)
        if what_is_match:
            return what_is_match.group(1).strip()
            
        # Check for "define X" pattern
        define_match = re.search(r"(?i)define ([^\?]+)", question)
        if define_match:
            return define_match.group(1).strip()
            
        # Fall back to longest entity
        if entities:
            return max(entities, key=len)
            
        return "the concept"
    
    def _extract_relevant_part(self, question: str, passage: str) -> str:
        """
        Extract most relevant part of passage based on question
        
        Args:
            question: The question text
            passage: The passage text
            
        Returns:
            Most relevant part of the passage
        """
        # Split into sentences
        sentences = self._split_into_sentences(passage)
        if not sentences:
            return passage
            
        # Extract key terms from question
        question_words = set(question.lower().split())
        
        # Score sentences by relevance
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            sentence_scores.append((sentence, overlap))
            
        # Sort by relevance score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 1-2 sentences
        if len(sentence_scores) >= 2 and sentence_scores[1][1] > 0:
            return " ".join([sentence_scores[0][0], sentence_scores[1][0]])
        
        return sentence_scores[0][0] if sentence_scores else passage
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _infer_question_type(self, question: str) -> str:
        """
        Infer the question type if not provided
        
        Args:
            question: The question text
            
        Returns:
            Inferred question type
        """
        question_lower = question.lower()
        
        # Check for definition questions
        if re.search(r"^what is|^what are|^who is|^who are|define|meaning of", question_lower):
            return "Definition"
            
        # Check for argument questions
        if re.search(r"^why|reason|cause|argument|justify", question_lower):
            return "Argument"
            
        # Check for context questions
        if re.search(r"^when|^where|context|history|period|time", question_lower):
            return "Context"
            
        # Check for evidence questions
        if re.search(r"evidence|example|support|prove", question_lower):
            return "Evidence"
            
        # Check for analogy questions
        if re.search(r"compare|similar|analogy|metaphor|parallel", question_lower):
            return "Analogy"
            
        # Default to Definition
        return "Definition"

def main():
    """Test the answer generator with example questions"""
    # Create example passages
    test_passage = {
        "text": "The history of all hitherto existing society is the history of class struggles. " 
                "Freeman and slave, patrician and plebeian, lord and serf, guild-master and journeyman, " 
                "in a word, oppressor and oppressed, stood in constant opposition to one another, " 
                "carried on an uninterrupted, now hidden, now open fight, a fight that each time ended, " 
                "either in a revolutionary reconstitution of society at large, or in the common ruin of the contending classes.",
        "score": 0.15,
        "keyword_score": 0.8,
        "hybrid_score": 0.9,
        "matched_keywords": ["class struggle", "revolution", "society"]
    }
    
    # Create answer generator
    generator = AnswerGenerator(use_llm=False)  # Don't use LLM for testing
    
    # Test questions and question types
    test_cases = [
        ("What is class struggle according to Marx?", "Definition"),
        ("Why does Marx analyze the conflict between classes?", "Argument"),
        ("When did Marx write about class struggle?", "Context"),
        ("What evidence does Marx provide for class conflict?", "Evidence"),
        ("How does Marx compare slaves and workers?", "Analogy")
    ]
    
    # Generate and print answers
    for question, question_type in test_cases:
        print(f"\nQuestion: {question}")
        print(f"Type: {question_type}")
        
        answer_data = generator.generate_answer(question, test_passage, question_type)
        
        print(f"Answer: {answer_data['answer']}")
        print(f"Confidence: {answer_data['confidence']:.2f}")
        print(f"Method: {answer_data['method']}")
        print("-" * 50)

def generate_answer_with_ollama(prompt: str, model: str = "mistral") -> Dict[str, Any]:
    """
    Generate an answer using Ollama
    
    Args:
        prompt: The prompt to send to Ollama
        model: The Ollama model to use
        
    Returns:
        Dictionary with generated answer and metadata
    """
    logger.info(f"Using Ollama with model {model}")
    
    try:
        # Prepare the request
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "system": "You are a helpful assistant that provides accurate, concise answers based on the passage text. Your answers should be factual and directly based on the provided information. Do not include HTML tags in your response.",
            "stream": False,
        }
        
        # Make the request
        response = requests.post(url, json=data)
        response.raise_for_status()
        response = response.json()
        
        # Get the response text and strip HTML tags
        response_text = response.get("response", "")
        clean_response = strip_html_tags(response_text)
        
        return {
            "answer": clean_response,
            "confidence": 0.9,
            "method": "ollama"
        }
        
    except Exception as e:
        logger.error(f"Error using Ollama: {str(e)}")
        return {
            "answer": f"Failed to generate answer with Ollama: {str(e)}",
            "confidence": 0.0,
            "method": "ollama_error"
        }

if __name__ == "__main__":
    main() 