#!/usr/bin/env python3
import os
import json
import argparse
import re
from typing import List, Dict, Any, Union

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    print("Transformers library not installed. Install with 'pip install transformers'")
    import sys
    sys.exit(1)

# Strong definition patterns (direct override)
definition_overrides = [
    r"^what is ", r"^what are ", r"^who is ", r"^who are ",
    r"^what does .+ mean", r"^define "
]

# Strong argument patterns (direct override)
argument_overrides = [
    r"^why ", r"^what reason", r"^for what reason", 
    r"what causes", r"what caused", r"how does .+ explain",
    r"what factors", r"what is the reasoning"
]

# Strong context patterns (direct override)
context_overrides = [
    r"^when was", r"^when did", r"^where was", r"^where did",
    r"^in what year", r"^during what period"
]

# Strong evidence patterns (direct override)
evidence_overrides = [
    r"^what evidence", r"^what examples", r"^what facts",
    r"^what data", r"^what observations"
]

# Strong analogy patterns (direct override)
analogy_overrides = [
    r"^how does .+ compare", r"^what similarities", 
    r"^how is .+ similar to", r"^what parallels", r"^how does .+ contrast"
]

def rule_based_classification(question: str) -> Union[str, None]:
    """
    Apply rule-based classification based on question patterns.
    Returns a classification if a rule matches, or None for fallback to ML model.
    
    Args:
        question: The question text to classify
        
    Returns:
        The category as a string, or None if no rule matches strongly
    """
    question_lower = question.lower()
    
    # Check for exact pattern matches with direct overrides
    for pattern in definition_overrides:
        if re.search(pattern, question_lower):
            return "Definition"
            
    for pattern in argument_overrides:
        if re.search(pattern, question_lower):
            return "Argument"
            
    for pattern in context_overrides:
        if re.search(pattern, question_lower):
            return "Context"
            
    for pattern in evidence_overrides:
        if re.search(pattern, question_lower):
            return "Evidence"
            
    for pattern in analogy_overrides:
        if re.search(pattern, question_lower):
            return "Analogy"
    
    # No strong rule match - return None for ML fallback
    return None

def classify_question(question: str, model_name: str = "facebook/bart-large-mnli") -> Union[str, Dict[str, Any]]:
    """
    Classify a question into one of the predefined categories using a hybrid approach:
    1. First try rule-based classification for high-confidence patterns
    2. Fall back to zero-shot classification for ambiguous cases
    
    Args:
        question: The question text to classify
        model_name: The Hugging Face model to use for zero-shot classification
                   Default is 'facebook/bart-large-mnli', other options include:
                   - 'roberta-large-mnli'
                   - 'distilbart-mnli-12-3'
                   - 'microsoft/deberta-v2-xlarge-mnli'
    
    Returns:
        Either just the predicted category as string (for backward compatibility)
        or a dict with category and scores if return_confidence=True
    """
    # Default confidence scores
    all_scores = {
        "Definition": 0.0,
        "Context": 0.0,
        "Argument": 0.0,
        "Evidence": 0.0,
        "Analogy": 0.0
    }
    
    # Track if we used rule-based classification
    used_rule_based = False
    rule_name = None
    
    # First, try rule-based classification
    rule_result = rule_based_classification(question)
    if rule_result:
        # We have a high-confidence rule match, mark it with a high confidence
        predicted_category = rule_result
        all_scores[predicted_category] = 0.99  # High confidence for rule-based match
        used_rule_based = True
        
        # Determine which rule was used
        question_lower = question.lower()
        
        # Check each pattern set to see which one matched
        if any(re.search(pattern, question_lower) for pattern in definition_overrides):
            rule_name = "definition_override"
        elif any(re.search(pattern, question_lower) for pattern in argument_overrides):
            rule_name = "argument_override"
        elif any(re.search(pattern, question_lower) for pattern in context_overrides):
            rule_name = "context_override"
        elif any(re.search(pattern, question_lower) for pattern in evidence_overrides):
            rule_name = "evidence_override"
        elif any(re.search(pattern, question_lower) for pattern in analogy_overrides):
            rule_name = "analogy_override"
        
        # Create a result object with all the metadata
        result = {
            "category": predicted_category,
            "confidence_scores": all_scores,
            "method": "rule_based",
            "rule_name": rule_name
        }
        
        return result
    
    # No strong rule match, continue with ML approach
    question_lower = question.lower()
    
    # Apply heuristic rules based on keywords and patterns
    
    # Check for Definition keywords
    definition_patterns = [
        r"^what is", r"^what are", r"^who is", r"^who are", 
        r"mean[s]?$", r"defin[e|es|ition]", r"describe"
    ]
    has_definition_pattern = any(re.search(pattern, question_lower) for pattern in definition_patterns)
    
    # Check for Argument keywords
    argument_patterns = [
        r"^why", r"justify", r"reason", r"thesis", r"argument", 
        r"how.*explain", r"principle", r"theoretical", r"theory",
        r"cause[s]?", r"factor[s]?", r"lead[s]? to", r"result[s]? in"
    ]
    has_argument_pattern = any(re.search(pattern, question_lower) for pattern in argument_patterns)
    
    # Check for Context keywords
    context_patterns = [
        r"^when", r"^where", r"period", r"during", r"historical", 
        r"published", r"written", r"time", r"background"
    ]
    has_context_pattern = any(re.search(pattern, question_lower) for pattern in context_patterns)
    
    # Check for Evidence keywords
    evidence_patterns = [
        r"example", r"evidence", r"fact", r"data", r"statistic", 
        r"observation", r"substantiate", r"support", r"prove"
    ]
    has_evidence_pattern = any(re.search(pattern, question_lower) for pattern in evidence_patterns)
    
    # Check for Analogy keywords
    analogy_patterns = [
        r"compar", r"similar", r"metaphor", r"analog", r"like", 
        r"parallel", r"resembl", r"contrasted", r"versus", r"vs"
    ]
    has_analogy_pattern = any(re.search(pattern, question_lower) for pattern in analogy_patterns)
    
    # Define the candidate categories with specific hypothesis templates
    category_templates = {
        "Definition": "This question is asking for the definition or meaning of a term or concept.",
        "Context": "This question is asking about historical context, background, or timing of events.",
        "Argument": "This question is asking about reasoning, justification, or thesis behind ideas.",
        "Evidence": "This question is asking for specific examples, data, or facts that support claims.",
        "Analogy": "This question is asking about comparisons, similarities, metaphors, or parallels between concepts or systems."
    }
    
    # Initialize the zero-shot classification pipeline with PyTorch
    # First load the model and tokenizer explicitly
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create the pipeline with explicit PyTorch components
    classifier = pipeline(
        "zero-shot-classification", 
        model=model, 
        tokenizer=tokenizer,
        framework="pt"  # Explicitly use PyTorch
    )
    
    # For multi-classification, combine all results with custom templates
    all_scores = {}
    raw_scores = {}
    
    # Run classification once for each category with its specific template
    for category, template in category_templates.items():
        result = classifier(
            question,
            candidate_labels=[category],
            hypothesis_template="This is a question. {}"
        )
        # Store the score for this category
        all_scores[category] = result["scores"][0]
        raw_scores[category] = result["scores"][0]  # Keep original scores
    
    # Apply stronger heuristic boosts based on pattern matches
    if has_definition_pattern:
        all_scores["Definition"] *= 1.5  # Increased from 1.3
    if has_argument_pattern:
        all_scores["Argument"] *= 1.5  # Increased from 1.3
    if has_context_pattern:
        all_scores["Context"] *= 1.2
    if has_evidence_pattern:
        all_scores["Evidence"] *= 1.2
    if has_analogy_pattern:
        all_scores["Analogy"] *= 1.3
    
    # Special case: "What causes" pattern should strongly bias toward Argument
    if re.search(r"what causes", question_lower) or re.search(r"what caused", question_lower):
        all_scores["Argument"] *= 2.0
        all_scores["Context"] *= 0.5  # Reduce Context score
    
    # Return the category with the highest score
    predicted_category = max(all_scores, key=all_scores.get)
    
    # Create a result object with all the metadata
    result = {
        "category": predicted_category,
        "confidence_scores": all_scores,
        "raw_scores": raw_scores,
        "method": "zero_shot",
        "applied_boosts": {
            "definition_pattern": has_definition_pattern,
            "argument_pattern": has_argument_pattern,
            "context_pattern": has_context_pattern,
            "evidence_pattern": has_evidence_pattern,
            "analogy_pattern": has_analogy_pattern,
            "what_causes_special_case": bool(re.search(r"what causes|what caused", question_lower))
        }
    }
    
    return result

def get_example_questions() -> Dict[str, List[str]]:
    """
    Return example questions for each category.
    
    Returns:
        A dictionary mapping category names to lists of example questions
    """
    return {
        "Definition": [
            "What is the bourgeoisie?",
            "How does Marx define class struggle?",
            "What does the term 'surplus value' mean in Marxist theory?",
            "What is the proletariat according to the Communist Manifesto?"
        ],
        "Context": [
            "When was the Communist Manifesto written?",
            "What historical events influenced Marx and Engels?",
            "What was happening in Europe when the Manifesto was published?",
            "How did industrialization affect class relations during Marx's time?"
        ],
        "Argument": [
            "Why does Marx believe capitalism will collapse?",
            "What is the main thesis of the Communist Manifesto?",
            "How does Marx justify the abolition of private property?",
            "What reasoning supports the idea of class consciousness?",
            "What causes revolution according to Marx?",  # Added this example
            "What factors lead to class conflict?"  # Added this example
        ],
        "Evidence": [
            "What examples does Marx provide for worker exploitation?",
            "What statistics or data does the Manifesto cite?",
            "What historical precedents support Marx's theory of revolution?",
            "What observations about industrial society substantiate Marx's claims?"
        ],
        "Analogy": [
            "How does Marx compare capitalism to feudalism?",
            "What metaphors does the Manifesto use to describe class struggle?",
            "How is the relationship between bourgeoisie and proletariat similar to other historical relationships?",
            "What parallels does Marx draw between economic and social structures?"
        ]
    }

def test_classification(output_file: str = "output/classification_tests.json") -> None:
    """
    Test the classification on example questions and save results to a JSON file.
    
    Args:
        output_file: Path to the output JSON file
    """
    examples = get_example_questions()
    results = []
    
    print("Testing question classification...")
    
    # Test all examples
    for category, questions in examples.items():
        for question in questions:
            result = classify_question(question)
            predicted = result["category"]
            confidence_scores = result["confidence_scores"]
            
            test_result = {
                "question": question,
                "expected": category,
                "predicted": predicted,
                "correct": predicted == category,
                "confidence_scores": confidence_scores,
                "method": result["method"]
            }
            
            # Add additional info based on method
            if result["method"] == "rule_based":
                test_result["rule_name"] = result["rule_name"]
            else:
                test_result["raw_scores"] = result["raw_scores"]
                test_result["applied_boosts"] = result["applied_boosts"]
            
            results.append(test_result)
            
            # Get confidence for predicted category
            confidence = confidence_scores[predicted]
            confidence_display = f"{confidence:.2f}"
            
            print(f"Question: {question}")
            print(f"Expected: {category}, Predicted: {predicted} (Confidence: {confidence_display})")
            print(f"Correct: {predicted == category}")
            print("---")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total
    
    # Calculate average confidence for correct and incorrect predictions
    confidence_correct = [r["confidence_scores"][r["predicted"]] for r in results if r["correct"]]
    confidence_incorrect = [r["confidence_scores"][r["predicted"]] for r in results if not r["correct"]]
    
    avg_confidence_correct = sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0
    avg_confidence_incorrect = sum(confidence_incorrect) / len(confidence_incorrect) if confidence_incorrect else 0
    
    summary = {
        "total_questions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "avg_confidence_correct": avg_confidence_correct,
        "avg_confidence_incorrect": avg_confidence_incorrect,
        "results": results
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Classification test results saved to {output_file}")
    print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")
    print(f"Avg Confidence (Correct): {avg_confidence_correct:.2f}")
    print(f"Avg Confidence (Incorrect): {avg_confidence_incorrect:.2f}")

def test_specific_patterns() -> None:
    """
    Run specific tests for the "What causes" pattern to verify it's correctly classified as Argument.
    """
    test_questions = [
        "What causes revolution?",
        "What causes class conflict?",
        "What causes economic crises according to Marx?",
        "What causes the fall of capitalism?",
        "What causes social change in Marxist theory?"
    ]
    
    print("\nTesting 'What causes' pattern classification...")
    print("=" * 50)
    
    all_correct = True
    for question in test_questions:
        result = classify_question(question)
        category = result["category"]
        confidence = result["confidence_scores"][category]
        method = result["method"]
        
        is_correct = category == "Argument"
        all_correct = all_correct and is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Question: {question}")
        print(f"  Predicted: {category} (Confidence: {confidence:.2f}, Method: {method})")
        print(f"  Expected: Argument")
        print(f"  Correct: {is_correct}")
        print("-" * 50)
    
    if all_correct:
        print("\nAll 'What causes' questions were correctly classified as Arguments! ✓")
    else:
        print("\nSome 'What causes' questions were not correctly classified! ✗")

def main():
    parser = argparse.ArgumentParser(description='Classify questions about the Communist Manifesto')
    parser.add_argument('--question', '-q', type=str, help='Question to classify')
    parser.add_argument('--model', '-m', default='facebook/bart-large-mnli',
                        help='Hugging Face model to use for classification')
    parser.add_argument('--test', '-t', action='store_true', help='Run tests on example questions')
    parser.add_argument('--output', '-o', default='output/classification_tests.json',
                        help='Output file for test results')
    parser.add_argument('--test-patterns', '-p', action='store_true', 
                        help='Test specific patterns like "What causes" questions')
    
    args = parser.parse_args()
    
    if args.test:
        test_classification(args.output)
    elif args.test_patterns:
        test_specific_patterns()
    elif args.question:
        result = classify_question(args.question, args.model)
        category = result["category"]
        confidence = result["confidence_scores"][category]
        method = result["method"]
        
        print(f"Question: {args.question}")
        print(f"Category: {category} (Confidence: {confidence:.2f}, Method: {method})")
        
        if method == "zero_shot":
            print("All confidence scores:")
            for cat, score in sorted(result["confidence_scores"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {score:.4f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 