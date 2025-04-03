#!/usr/bin/env python3
import os
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
import pandas as pd

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:
    print("spaCy not installed. Install with 'pip install spacy' and 'python -m spacy download en_core_web_sm'")
    sys.exit(1)

# Define key Marxist concepts and their related terms
CORE_CONCEPTS = {
    "class struggle": ["class antagonism", "class conflict", "class warfare", "struggle between classes"],
    "bourgeoisie": ["bourgeois", "capitalists", "middle class", "property owners"],
    "proletariat": ["proletarian", "working class", "laborer", "working men", "laborers", "wage laborers"],
    "means of production": ["productive forces", "instruments of production", "machinery", "industrial capital"],
    "capitalism": ["capitalist mode of production", "capitalist system", "free competition", "market economy"],
    "communism": ["communist", "communistic", "common ownership", "socialised production"],
    "socialism": ["socialist", "social ownership", "public ownership"],
    "exploitation": ["exploitation of labor", "exploitation of the proletariat", "oppression"],
    "alienation": ["estrangement", "alienated labor", "alienated from labor"],
    "revolution": ["revolutionary", "revolt", "uprising", "overthrow", "class revolution"],
    "historical materialism": ["materialist conception of history", "economic determinism"],
    "dialectic": ["dialectical", "contradiction", "thesis", "antithesis", "synthesis"],
    "class consciousness": ["class awareness", "political consciousness"],
    "commodity": ["commodities", "commodity production", "commodity exchange", "commodification"],
    "surplus value": ["surplus labor", "unpaid labor", "profit"],
    "primitive accumulation": ["original accumulation", "expropriation", "enclosure"],
    "imperialism": ["colonialism", "colonial system", "expansion of capital"],
    "ideology": ["ideological", "false consciousness", "ruling ideas"]
}

def load_jsonl(file_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def tag_concepts(data, nlp, output_file):
    """
    Tag abstract concepts in text data.
    
    Args:
        data: List of dictionaries with text data
        nlp: spaCy language model
        output_file: Path to save concept dictionary
    
    Returns:
        Updated data with concept tags
    """
    # Create concept matchers
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    concept_patterns = {}
    
    for concept, terms in CORE_CONCEPTS.items():
        patterns = [nlp.make_doc(term) for term in [concept] + terms]
        matcher.add(concept, patterns)
        concept_patterns[concept] = terms
    
    # Initialize concept counts
    concept_counts = Counter()
    concept_contexts = defaultdict(list)
    
    # Process texts and find concepts
    for i, item in enumerate(data):
        text = item['text']
        doc = nlp(text)
        
        # Find matches
        matches = matcher(doc)
        item_concepts = set()
        
        for match_id, start, end in matches:
            concept = nlp.vocab.strings[match_id]
            span = doc[start:end].text
            item_concepts.add(concept)
            
            # Store context for this concept
            context = {
                'paragraph_id': item['paragraph_id'],
                'section': item['section'],
                'text': text,
                'matched_term': span
            }
            concept_contexts[concept].append(context)
        
        # Add concepts to the item
        data[i]['concepts'] = list(item_concepts)
        
        # Update counts
        concept_counts.update(item_concepts)
    
    # Create concept dictionary
    concept_dict = []
    for concept, count in concept_counts.most_common():
        contexts = concept_contexts[concept]
        concept_dict.append({
            'concept': concept,
            'count': count,
            'related_terms': concept_patterns[concept],
            'sample_contexts': contexts[:5]  # Store up to 5 example contexts
        })
    
    # Save concept dictionary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(concept_dict, f, indent=2)
    
    print(f"Saved concept dictionary with {len(concept_dict)} concepts to {output_file}")
    
    # Also save as CSV for easy viewing
    csv_path = os.path.splitext(output_file)[0] + '.csv'
    concept_df = pd.DataFrame([{
        'concept': c['concept'], 
        'count': c['count'],
        'related_terms': ', '.join(c['related_terms'])
    } for c in concept_dict])
    
    concept_df.to_csv(csv_path, index=False)
    print(f"Saved simplified concept CSV to {csv_path}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Tag theoretical concepts in The Communist Manifesto')
    parser.add_argument('input_file', help='Path to the processed JSONL file')
    parser.add_argument('--output-file', '-o', default='data/manifesto_with_concepts.jsonl',
                        help='Path to save output JSONL file with concepts')
    parser.add_argument('--concept-dict', '-d', default='data/concept_dictionary.json',
                        help='Path to save concept dictionary')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Please download it with:")
        print("python -m spacy download en_core_web_sm")
        return 1
    
    # Load data
    try:
        data = load_jsonl(args.input_file)
        print(f"Loaded {len(data)} records from {args.input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.concept_dict), exist_ok=True)
    
    # Process data
    try:
        updated_data = tag_concepts(data, nlp, args.concept_dict)
        
        # Save updated data
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in updated_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(updated_data)} records with concept tags to {args.output_file}")
    except Exception as e:
        print(f"Error processing data: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 