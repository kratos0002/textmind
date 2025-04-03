#!/usr/bin/env python3
"""
Vector-based semantic search for 'The Communist Manifesto'
Simplified implementation using numpy and scipy for vector operations
"""

import os
import json
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModel
import torch

# Directory constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")

# Ensure directories exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)

class TextRetriever:
    """Text retrieval system using vector-based semantic search"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the retriever with the specified transformer model
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.paragraphs = []
        self.paragraph_embeddings = None
        self.concept_dict = None
        
    def load_paragraphs(self, jsonl_path: str = os.path.join(DATA_DIR, "manifesto_clean.jsonl")) -> None:
        """
        Load paragraphs from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file containing text paragraphs
        """
        self.paragraphs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    self.paragraphs.append(data)
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line}")
        
        print(f"Loaded {len(self.paragraphs)} paragraphs")
    
    def load_concept_dictionary(self, csv_path: str = os.path.join(DATA_DIR, "concept_dictionary.csv")) -> None:
        """
        Load concept dictionary for potential reranking
        
        Args:
            csv_path: Path to CSV file containing concept dictionary
        """
        self.concept_dict = pd.read_csv(csv_path)
        # Expand related terms into a mapping
        self.concept_term_mapping = {}
        for _, row in self.concept_dict.iterrows():
            concept = row['concept']
            if pd.notna(row['related_terms']):
                terms = [term.strip() for term in row['related_terms'].split(',')]
                terms.append(concept)  # Add the main concept itself
                for term in terms:
                    self.concept_term_mapping[term.lower()] = concept
        
        print(f"Loaded {len(self.concept_dict)} concepts with {len(self.concept_term_mapping)} related terms")
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenization
            
        Returns:
            Mean-pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_text(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                           max_length=512, return_tensors='pt')
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Apply mean pooling
            batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings.cpu().numpy()
    
    def create_embeddings(self) -> None:
        """Generate embeddings for all paragraphs"""
        texts = [p['text'] for p in self.paragraphs]
        self.paragraph_embeddings = self.encode_text(texts)
        print(f"Created embeddings with shape {self.paragraph_embeddings.shape}")
    
    def save_embeddings(self, embeddings_path: str = os.path.join(EMBEDDING_DIR, "paragraph_embeddings.npy")) -> None:
        """
        Save embeddings to disk
        
        Args:
            embeddings_path: Path to save the embeddings
        """
        if self.paragraph_embeddings is None:
            raise ValueError("Please create embeddings first using create_embeddings()")
        
        # Save embeddings
        np.save(embeddings_path, self.paragraph_embeddings)
        
        # Save paragraph metadata
        metadata_path = os.path.join(EMBEDDING_DIR, "paragraph_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # Save minimal information needed for retrieval
            metadata = [
                {
                    "section": p["section"],
                    "paragraph_index": p["paragraph_index"],
                    "paragraph_id": p["paragraph_id"]
                }
                for p in self.paragraphs
            ]
            json.dump(metadata, f, indent=2)
            
        print(f"Saved embeddings to {embeddings_path} and metadata to {metadata_path}")
    
    def load_embeddings(self, embeddings_path: str = os.path.join(EMBEDDING_DIR, "paragraph_embeddings.npy")) -> None:
        """
        Load embeddings from disk
        
        Args:
            embeddings_path: Path to load the embeddings from
        """
        self.paragraph_embeddings = np.load(embeddings_path)
        
        # Load the full paragraphs if they're not already loaded
        if not self.paragraphs:
            self.load_paragraphs()
            
        print(f"Loaded embeddings with shape {self.paragraph_embeddings.shape}")
    
    def retrieve_relevant_passages(self, query: str, top_k: int = 3, 
                                   use_reranking: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant paragraphs for the query
        
        Args:
            query: The search query
            top_k: Number of results to return
            use_reranking: Whether to apply concept-based reranking
            
        Returns:
            List of dictionaries containing the relevant paragraphs with metadata
        """
        if self.paragraph_embeddings is None:
            raise ValueError("Embeddings have not been created or loaded yet")
        
        # Encode query
        query_embedding = self.encode_text([query])[0]
        
        # Calculate distances
        distances = cdist([query_embedding], self.paragraph_embeddings, metric='cosine')[0]
        
        # Get top results (indices sorted by distance)
        indices = np.argsort(distances)[:min(top_k * 2, len(self.paragraphs))]
        
        # Get results
        results = []
        for idx in indices:
            paragraph = self.paragraphs[idx]
            results.append({
                "section": paragraph["section"],
                "paragraph_index": paragraph["paragraph_index"],
                "paragraph_id": paragraph["paragraph_id"],
                "text": paragraph["text"],
                "score": float(distances[idx])
            })
        
        # Apply concept-based reranking if requested
        if use_reranking and self.concept_dict is not None:
            # Extract concepts from query
            query_concepts = self._extract_concepts_from_text(query)
            
            # Rerank based on concept overlap
            for result in results:
                paragraph_concepts = self._extract_concepts_from_text(result["text"])
                
                # Calculate overlap score
                overlap = len(set(query_concepts) & set(paragraph_concepts))
                
                # Adjust the score based on concept overlap
                # Lower scores are better in cosine distance, so we subtract
                result["score"] = result["score"] - (overlap * 0.05)  # Weight factor for concepts
                
                # Add concepts for debugging/explanation
                result["matching_concepts"] = list(set(query_concepts) & set(paragraph_concepts))
            
            # Re-sort by adjusted score
            results = sorted(results, key=lambda x: x["score"])
        
        # Return top-k after potential reranking
        return results[:top_k]
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """
        Extract known concepts from text using the concept dictionary
        
        Args:
            text: Text to extract concepts from
            
        Returns:
            List of concepts found in the text
        """
        if self.concept_dict is None:
            return []
        
        found_concepts = set()
        text_lower = text.lower()
        
        # Look for each term in the text
        for term, concept in self.concept_term_mapping.items():
            if term in text_lower:
                found_concepts.add(concept)
        
        return list(found_concepts)


def main():
    """Main function to demonstrate the retrieval system"""
    print("TextMind Semantic Search")
    print("=======================")
    
    # Initialize retriever
    retriever = TextRetriever()
    
    # Check if embeddings exist
    embeddings_path = os.path.join(EMBEDDING_DIR, "paragraph_embeddings.npy")
    if os.path.exists(embeddings_path):
        print("Loading existing embeddings...")
        retriever.load_embeddings(embeddings_path)
    else:
        print("Creating new embeddings...")
        retriever.load_paragraphs()
        retriever.create_embeddings()
        retriever.save_embeddings(embeddings_path)
    
    # Load concept dictionary for reranking
    try:
        retriever.load_concept_dictionary()
    except Exception as e:
        print(f"Could not load concept dictionary: {e}")
    
    # Example queries
    example_queries = [
        "What is the relationship between the bourgeoisie and the proletariat?",
        "How does Marx define communism?",
        "What historical role does the Communist Party play?"
    ]
    
    # Demonstrate retrieval
    for query in example_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Get results with reranking
        results = retriever.retrieve_relevant_passages(query, top_k=3, use_reranking=True)
        
        # Display results
        for i, result in enumerate(results):
            print(f"Result {i+1} (Section: {result['section']}, Para: {result['paragraph_index']})")
            print(f"Score: {result['score']:.4f}")
            if 'matching_concepts' in result and result['matching_concepts']:
                print(f"Matching concepts: {', '.join(result['matching_concepts'])}")
            
            # Print a snippet of the text (first 200 chars)
            text_snippet = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"Snippet: {text_snippet}")
            print()


if __name__ == "__main__":
    main() 