#!/usr/bin/env python3
"""
Self-evaluation test pipeline for TextMind Q&A system.

This script evaluates the performance of the question classifier and retriever
on a sample set of questions with expected keywords.
"""

import os
import sys
import json
from typing import List, Dict, Any
from colorama import init, Fore, Style

# Add parent directory to path to allow importing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our custom modules
from scripts.question_classifier import classify_question
from scripts.retriever import TextRetriever
from scripts.reranker import rerank_passages  # Import the new reranker

# Initialize colorama for colored terminal output
init()

# Sample test questions with expected keywords
test_questions = [
    {
        "question": "What is class struggle?",
        "expected_keywords": ["class", "struggle", "oppressor", "oppressed"],
        "expected_type": "Definition"
    },
    {
        "question": "Why does Marx criticize the bourgeoisie?",
        "expected_keywords": ["bourgeoisie", "capital", "exploitation"],
        "expected_type": "Argument"
    },
    {
        "question": "What causes revolution?",
        "expected_keywords": ["revolution", "class", "proletariat", "conflict"],
        "expected_type": "Argument"
    }
]

def check_keyword_coverage(passage_text: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Check the coverage of expected keywords in a passage
    
    Args:
        passage_text: The text to check
        keywords: List of expected keywords
    
    Returns:
        Dictionary with matched keywords, count, and coverage score
    """
    passage_lower = passage_text.lower()
    matched_keywords = []
    
    for keyword in keywords:
        if keyword.lower() in passage_lower:
            matched_keywords.append(keyword)
    
    coverage = len(matched_keywords) / len(keywords) if keywords else 0
    
    return {
        "matched_keywords": matched_keywords,
        "matched_count": len(matched_keywords),
        "total_keywords": len(keywords),
        "coverage": coverage
    }

def is_passage_relevant(passage_text: str, question: str) -> bool:
    """
    Basic relevance check - looks for question terms in the passage
    
    Args:
        passage_text: The retrieved passage
        question: The original question
    
    Returns:
        Boolean indicating if the passage seems relevant
    """
    # Extract key nouns from question (simple approach)
    question_words = question.lower().split()
    question_words = [w for w in question_words if len(w) > 3 and w not in 
                     ["what", "why", "how", "does", "did", "has", "have", "the", "and", "that"]]
    
    passage_lower = passage_text.lower()
    matched_terms = sum(1 for word in question_words if word in passage_lower)
    
    # Consider relevant if at least half of question terms are in passage
    return matched_terms >= len(question_words) / 2

def generate_score_icon(score: float) -> str:
    """Generate a visual indicator for the score"""
    if score >= 0.8:
        return f"{Fore.GREEN}✅ Great{Style.RESET_ALL}"
    elif score >= 0.5:
        return f"{Fore.YELLOW}⚠️ Fair{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}❌ Poor{Style.RESET_ALL}"

def generate_feedback(question_type_correct: bool, keyword_coverage: float, relevance: bool) -> str:
    """Generate feedback based on evaluation metrics"""
    feedback = []
    
    if not question_type_correct:
        feedback.append("Question classification may need improvement")
    
    if not relevance:
        feedback.append("Retrieved passage seems off-topic")
    
    if keyword_coverage < 0.5:
        feedback.append("Low keyword coverage, consider improving retrieval")
    elif keyword_coverage < 0.8:
        feedback.append("Moderate keyword coverage, consider reranking for stronger matches")
    else:
        feedback.append("Good keyword coverage")
    
    return "; ".join(feedback)

def evaluate_system() -> Dict[str, Any]:
    """
    Run the evaluation on the test questions
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"{Fore.CYAN}TextMind Self-Evaluation Test Pipeline{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    
    # Initialize retriever
    retriever = TextRetriever()
    
    # Load embeddings if they exist
    embeddings_path = os.path.join(parent_dir, "output", "embeddings", "paragraph_embeddings.npy")
    
    if os.path.exists(embeddings_path):
        print(f"\nLoading existing embeddings from {embeddings_path}")
        retriever.load_embeddings(embeddings_path)
        
        # Load concept dictionary for reranking
        try:
            retriever.load_concept_dictionary()
            print("Loaded concept dictionary for reranking")
        except Exception as e:
            print(f"Warning: Could not load concept dictionary: {e}")
    else:
        print(f"\n{Fore.RED}Error: Embeddings not found. Please run scripts/retriever.py first.{Style.RESET_ALL}")
        sys.exit(1)
    
    # Track overall results
    results = {
        "questions": [],
        "metrics": {
            "question_type_accuracy": 0,
            "keyword_coverage": 0,
            "relevance_rate": 0,
            "overall_score": 0
        }
    }
    
    # Track results with and without custom reranker
    reranker_results = {
        "question_type_accuracy": 0,
        "keyword_coverage": 0,
        "relevance_rate": 0,
        "overall_score": 0
    }
    
    # Process each test question
    total_questions = len(test_questions)
    correct_types = 0
    total_coverage = 0
    total_relevance = 0
    
    reranker_total_coverage = 0
    reranker_total_relevance = 0
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        expected_type = test_case.get("expected_type", "")  # Optional
        
        print(f"\n{Fore.YELLOW}Question {i}/{total_questions}:{Style.RESET_ALL} {question}")
        print("-" * 60)
        
        # Step 1: Classify the question
        predicted_type = classify_question(question)
        type_correct = expected_type == "" or predicted_type == expected_type
        if type_correct:
            correct_types += 1
        
        type_result = f"{Fore.GREEN}Correct{Style.RESET_ALL}" if type_correct else f"{Fore.RED}Incorrect (Expected: {expected_type}){Style.RESET_ALL}"
        print(f"Predicted Type: {predicted_type} - {type_result}")
        
        # Step 2: Retrieve passages
        passages = retriever.retrieve_relevant_passages(
            query=question,
            top_k=5,  # Retrieve more for better reranking chances
            use_reranking=True
        )
        
        # Step 3: Apply our custom reranker
        reranked_passages = rerank_passages(question, passages, expected_keywords)
        
        # Step 4: Evaluate both original and reranked passages
        passage_result = {
            "question": question,
            "expected_type": expected_type,
            "predicted_type": predicted_type,
            "type_correct": type_correct,
            "original_passages": [],
            "reranked_passages": []
        }
        
        # Evaluate original top passage
        original_top_coverage = 0
        original_relevance = False
        
        if passages:
            keyword_check = check_keyword_coverage(passages[0]["text"], expected_keywords)
            is_relevant = is_passage_relevant(passages[0]["text"], question)
            original_top_coverage = keyword_check["coverage"]
            original_relevance = is_relevant
            if original_relevance:
                total_relevance += 1
        
        # Evaluate reranked top passage
        reranked_top_coverage = 0
        reranked_relevance = False
        
        if reranked_passages:
            reranked_keyword_check = check_keyword_coverage(reranked_passages[0]["text"], expected_keywords)
            reranked_is_relevant = is_passage_relevant(reranked_passages[0]["text"], question)
            reranked_top_coverage = reranked_keyword_check["coverage"]
            reranked_relevance = reranked_is_relevant
            if reranked_relevance:
                reranker_total_relevance += 1
            
            # Add the results for reranked passages
            reranker_total_coverage += reranked_top_coverage
        
        # Print comparison between original and reranked results
        print(f"\n{Fore.CYAN}Comparison of Retrieval Methods:{Style.RESET_ALL}")
        print(f"Original Top Passage Keyword Coverage: {original_top_coverage:.2f}")
        print(f"Reranked Top Passage Keyword Coverage: {reranked_top_coverage:.2f}")
        
        coverage_diff = reranked_top_coverage - original_top_coverage
        if coverage_diff > 0:
            print(f"Improvement with Reranker: {Fore.GREEN}+{coverage_diff:.2f}{Style.RESET_ALL}")
        elif coverage_diff < 0:
            print(f"Change with Reranker: {Fore.RED}{coverage_diff:.2f}{Style.RESET_ALL}")
        else:
            print(f"No change in keyword coverage")
        
        # Display detailed passage information (using reranked passages)
        for j, passage in enumerate(reranked_passages, 1):
            # Check keyword coverage
            keyword_check = check_keyword_coverage(passage["text"], expected_keywords)
            
            # Check if passage is relevant
            is_relevant = is_passage_relevant(passage["text"], question)
            
            # Add passage results
            passage_result["reranked_passages"].append({
                "rank": j,
                "text": passage["text"],
                "section": passage["section"],
                "paragraph_index": passage["paragraph_index"],
                "keyword_coverage": keyword_check["coverage"],
                "matched_keywords": keyword_check["matched_keywords"],
                "is_relevant": is_relevant
            })
            
            # Print passage evaluation
            relevance_text = f"{Fore.GREEN}Relevant{Style.RESET_ALL}" if is_relevant else f"{Fore.RED}Possibly Off-Topic{Style.RESET_ALL}"
            
            print(f"\nReranked Passage {j}: (Section: {passage['section']}, Para: {passage['paragraph_index']})")
            text_preview = passage["text"][:200] + "..." if len(passage["text"]) > 200 else passage["text"]
            print(f"Text: {text_preview}")
            
            if 'matching_concepts' in passage and passage['matching_concepts']:
                print(f"Matching concepts: {', '.join(passage['matching_concepts'])}")
            
            print(f"Keyword Match: {keyword_check['matched_count']}/{keyword_check['total_keywords']} " + 
                 f"({', '.join(keyword_check['matched_keywords'])})")
            
            print(f"Relevance: {relevance_text}")
            print(f"Coverage Score: {keyword_check['coverage']:.2f}")
            
            # If this passage has a keyword_score from our reranker, show it
            if 'keyword_score' in passage:
                print(f"Reranker Score: {passage['keyword_score']:.2f}")
        
        # Calculate aggregate score for this question (70% keyword coverage, 30% correct type)
        original_question_score = (0.7 * original_top_coverage) + (0.3 * int(type_correct))
        reranked_question_score = (0.7 * reranked_top_coverage) + (0.3 * int(type_correct))
        
        score_icon = generate_score_icon(reranked_question_score)
        
        # Generate feedback
        feedback = generate_feedback(type_correct, reranked_top_coverage, reranked_relevance)
        
        print(f"\n{Fore.CYAN}ORIGINAL QUESTION SCORE: {original_question_score:.2f}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}RERANKED QUESTION SCORE: {reranked_question_score:.2f} {score_icon}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}FEEDBACK: {feedback}{Style.RESET_ALL}")
        
        # Add to results
        passage_result["original_score"] = original_question_score
        passage_result["reranked_score"] = reranked_question_score
        passage_result["feedback"] = feedback
        results["questions"].append(passage_result)
        
        # Update overall metrics
        total_coverage += original_top_coverage
    
    # Calculate original metrics
    results["metrics"]["question_type_accuracy"] = correct_types / total_questions
    results["metrics"]["keyword_coverage"] = total_coverage / total_questions
    results["metrics"]["relevance_rate"] = total_relevance / total_questions
    results["metrics"]["overall_score"] = (0.3 * results["metrics"]["question_type_accuracy"] + 
                                          0.4 * results["metrics"]["keyword_coverage"] +
                                          0.3 * results["metrics"]["relevance_rate"])
    
    # Calculate reranker metrics
    reranker_results["question_type_accuracy"] = correct_types / total_questions
    reranker_results["keyword_coverage"] = reranker_total_coverage / total_questions
    reranker_results["relevance_rate"] = reranker_total_relevance / total_questions
    reranker_results["overall_score"] = (0.3 * reranker_results["question_type_accuracy"] + 
                                        0.4 * reranker_results["keyword_coverage"] +
                                        0.3 * reranker_results["relevance_rate"])
    
    # Add reranker results to the results dictionary
    results["reranker_metrics"] = reranker_results
    
    # Print overall results with comparison
    print(f"\n{Fore.CYAN}======================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}OVERALL EVALUATION RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    
    # Original metrics
    print(f"{Fore.YELLOW}Original Retriever:{Style.RESET_ALL}")
    print(f"Question Type Accuracy: {results['metrics']['question_type_accuracy']:.2f}")
    print(f"Keyword Coverage: {results['metrics']['keyword_coverage']:.2f}")
    print(f"Relevance Rate: {results['metrics']['relevance_rate']:.2f}")
    print(f"Overall Score: {results['metrics']['overall_score']:.2f} {generate_score_icon(results['metrics']['overall_score'])}")
    
    # Reranker metrics
    print(f"\n{Fore.YELLOW}With Custom Reranker:{Style.RESET_ALL}")
    print(f"Question Type Accuracy: {reranker_results['question_type_accuracy']:.2f}")
    print(f"Keyword Coverage: {reranker_results['keyword_coverage']:.2f}")
    print(f"Relevance Rate: {reranker_results['relevance_rate']:.2f}")
    print(f"Overall Score: {reranker_results['overall_score']:.2f} {generate_score_icon(reranker_results['overall_score'])}")
    
    # Calculate and show improvement
    improvement = reranker_results['overall_score'] - results['metrics']['overall_score']
    if improvement > 0:
        print(f"\n{Fore.GREEN}Improvement with Reranker: +{improvement:.2f} ({improvement*100:.1f}%){Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Change with Reranker: {improvement:.2f} ({improvement*100:.1f}%){Style.RESET_ALL}")
    
    # Generate system improvement suggestions
    print(f"\n{Fore.YELLOW}SYSTEM IMPROVEMENT SUGGESTIONS:{Style.RESET_ALL}")
    
    if reranker_results["question_type_accuracy"] < 0.7:
        print("- Tune question classifier to better match expected types")
    
    if reranker_results["keyword_coverage"] < 0.7:
        print("- Improve embedding model or switch to a more domain-specific model")
        print("- Further enhance keyword-based reranking algorithms")
    
    if reranker_results["relevance_rate"] < 0.7:
        print("- Add contextual relevance scoring")
        print("- Explore query expansion to improve matching")
    
    # Save results to file
    output_file = os.path.join(parent_dir, "output", "evaluation_results.json")
    
    # Convert results to serializable format (remove ANSI color codes)
    clean_results = json.loads(json.dumps(results, default=str))
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    evaluate_system() 