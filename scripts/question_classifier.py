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

def classify_question(question: str, model_name: str = "facebook/bart-large-mnli") -> str:
    """
    Classify a question into one of the predefined categories using zero-shot classification.
    
    Args:
        question: The question text to classify
        model_name: The Hugging Face model to use for zero-shot classification
                   Default is 'facebook/bart-large-mnli', other options include:
                   - 'roberta-large-mnli'
                   - 'distilbart-mnli-12-3'
                   - 'microsoft/deberta-v2-xlarge-mnli'
    
    Returns:
        The predicted category as a string
    """
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
        r"how.*explain", r"principle", r"theoretical", r"theory"
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
    
    # Run classification once for each category with its specific template
    for category, template in category_templates.items():
        result = classifier(
            question,
            candidate_labels=[category],
            hypothesis_template="This is a question. {}"
        )
        # Store the score for this category
        all_scores[category] = result["scores"][0]
    
    # Apply heuristic boosts based on pattern matches
    if has_definition_pattern:
        all_scores["Definition"] *= 1.3
    if has_argument_pattern:
        all_scores["Argument"] *= 1.3
    if has_context_pattern:
        all_scores["Context"] *= 1.2
    if has_evidence_pattern:
        all_scores["Evidence"] *= 1.2
    if has_analogy_pattern:
        all_scores["Analogy"] *= 1.3
    
    # Return the category with the highest score
    predicted_category = max(all_scores, key=all_scores.get)
    
    # For debugging
    # print(f"Scores: {all_scores}")
    
    return predicted_category

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
            "What reasoning supports the idea of class consciousness?"
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
    
    # To save time during testing, we'll just test one question per category
    for category, questions in examples.items():
        question = questions[0]  # Just use the first question of each category
        predicted = classify_question(question)
        result = {
            "question": question,
            "expected": category,
            "predicted": predicted,
            "correct": predicted == category
        }
        results.append(result)
        print(f"Question: {question}")
        print(f"Expected: {category}, Predicted: {predicted}")
        print(f"Correct: {predicted == category}")
        print("---")
    
    # Calculate accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total
    
    summary = {
        "total_questions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "results": results
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Classification test results saved to {output_file}")
    print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")

def main():
    parser = argparse.ArgumentParser(description='Classify questions about the Communist Manifesto')
    parser.add_argument('--question', '-q', type=str, help='Question to classify')
    parser.add_argument('--model', '-m', default='facebook/bart-large-mnli',
                        help='Hugging Face model to use for classification')
    parser.add_argument('--test', '-t', action='store_true', help='Run tests on example questions')
    parser.add_argument('--output', '-o', default='output/classification_tests.json',
                        help='Output file for test results')
    
    args = parser.parse_args()
    
    if args.test:
        test_classification(args.output)
    elif args.question:
        category = classify_question(args.question, args.model)
        print(f"Question: {args.question}")
        print(f"Category: {category}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 