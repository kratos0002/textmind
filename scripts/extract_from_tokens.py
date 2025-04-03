#!/usr/bin/env python3
import os
import re
import json
import pandas as pd
import argparse
import sys

def identify_sections(tokens_df):
    """
    Identify section titles based on patterns in The Communist Manifesto
    """
    # Group tokens by sentence_ID
    sentences = tokens_df.groupby('sentence_ID').agg({
        'word': lambda x: ' '.join(x),
        'paragraph_ID': 'first'
    }).reset_index()
    
    # Identify potential section titles 
    # For Communist Manifesto, sections typically look like "I. BOURGEOIS AND PROLETARIANS"
    section_pattern = re.compile(r'^(?:(?:I{1,4}|V?I{0,3}|IX|X)\.?)[\s\.].*')
    
    # Mark section titles
    sentences['is_section_title'] = sentences['word'].apply(
        lambda x: bool(section_pattern.match(x.strip().upper()))
    )
    
    # Create section mapping
    sections = []
    current_section = "Preface"
    section_para_counter = 0
    
    section_mapping = {}
    for _, row in sentences.iterrows():
        if row['is_section_title']:
            current_section = row['word'].strip()
            section_para_counter = 0
            # Add to our list of sections
            if current_section not in sections:
                sections.append(current_section)
        else:
            section_para_counter += 1
        
        section_mapping[row['sentence_ID']] = {
            'section': current_section,
            'paragraph_ID': row['paragraph_ID'],
            'paragraph_index': section_para_counter
        }
    
    print(f"Identified {len(sections)} sections")
    return section_mapping, sections

def process_supersense(supersense_file, tokens_df):
    """
    Process supersense tags to extract semantic information
    """
    supersense_df = pd.read_csv(supersense_file, sep='\t')
    
    # Merge supersense data with tokens
    supersense_concepts = {}
    for _, row in supersense_df.iterrows():
        start_token = row['start_token']
        end_token = row['end_token']
        supersense = row['supersense_category']
        
        # Get the actual text
        token_slice = tokens_df[(tokens_df['token_ID_within_document'] >= start_token) & 
                               (tokens_df['token_ID_within_document'] <= end_token)]
        
        text = ' '.join(token_slice['word'].tolist())
        
        # Save to dictionary
        if supersense not in supersense_concepts:
            supersense_concepts[supersense] = []
        
        supersense_concepts[supersense].append(text)
    
    return supersense_concepts

def extract_paragraphs(tokens_df, section_mapping):
    """
    Extract paragraphs with section information
    """
    paragraphs = tokens_df.groupby('paragraph_ID').agg({
        'word': lambda x: ' '.join(x),
        'sentence_ID': lambda x: list(set(x))
    }).reset_index()
    
    # Add section information
    paragraphs['section'] = paragraphs['sentence_ID'].apply(
        lambda sent_ids: section_mapping[sent_ids[0]]['section'] if sent_ids else "Unknown"
    )
    
    paragraphs['paragraph_index'] = paragraphs['sentence_ID'].apply(
        lambda sent_ids: section_mapping[sent_ids[0]]['paragraph_index'] if sent_ids else 0
    )
    
    # Clean text
    paragraphs['text'] = paragraphs['word'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    # Keep only useful columns
    paragraphs = paragraphs[['paragraph_ID', 'section', 'paragraph_index', 'text']]
    
    return paragraphs

def save_to_jsonl(records, output_file):
    """
    Save records to JSONL format
    """
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved {len(records)} records to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract structured data from BookNLP tokens')
    parser.add_argument('--tokens-file', '-t', default='output/booknlp/manifesto.tokens',
                        help='Path to the tokens file')
    parser.add_argument('--supersense-file', '-s', default='output/booknlp/manifesto.supersense',
                        help='Path to the supersense file')
    parser.add_argument('--output-file', '-o', default='data/manifesto_clean.jsonl',
                        help='Path to save the output JSONL file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.tokens_file):
        print(f"Error: Tokens file not found: {args.tokens_file}")
        return 1
    
    if not os.path.exists(args.supersense_file):
        print(f"Error: Supersense file not found: {args.supersense_file}")
        return 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    try:
        # Load tokens
        print(f"Loading tokens from {args.tokens_file}")
        tokens_df = pd.read_csv(args.tokens_file, sep='\t')
        
        # Identify sections
        print("Identifying sections...")
        section_mapping, sections = identify_sections(tokens_df)
        
        # Process supersense
        print(f"Processing supersense data from {args.supersense_file}")
        supersense_concepts = process_supersense(args.supersense_file, tokens_df)
        
        # Extract paragraphs
        print("Extracting paragraphs...")
        paragraphs = extract_paragraphs(tokens_df, section_mapping)
        
        # Create output records
        output_records = []
        for _, row in paragraphs.iterrows():
            record = {
                'section': row['section'],
                'paragraph_index': int(row['paragraph_index']),
                'paragraph_id': int(row['paragraph_ID']),
                'text': row['text']
            }
            output_records.append(record)
        
        # Save to JSONL
        save_to_jsonl(output_records, args.output_file)
        
        # Also save sections list and supersense concepts for knowledge graph creation
        sections_file = os.path.join(os.path.dirname(args.output_file), 'manifesto_sections.json')
        with open(sections_file, 'w') as f:
            json.dump(sections, f, indent=2)
        print(f"Saved sections to {sections_file}")
        
        concepts_file = os.path.join(os.path.dirname(args.output_file), 'manifesto_concepts.json')
        with open(concepts_file, 'w') as f:
            json.dump(supersense_concepts, f, indent=2)
        print(f"Saved supersense concepts to {concepts_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 