#!/usr/bin/env python3
"""
Test script to verify concept data loading and availability
"""

import json
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("concept_test")

def load_concept_data(concept_dict_path="data/concept_dictionary.json"):
    """Load concept dictionary from JSON file"""
    try:
        logger.info(f"Attempting to load concepts from {concept_dict_path}")
        with open(concept_dict_path, 'r', encoding='utf-8') as f:
            concepts = json.load(f)
        logger.info(f"Loaded {len(concepts)} concepts from {concept_dict_path}")
        return concepts
    except FileNotFoundError:
        logger.error(f"File not found: {concept_dict_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {concept_dict_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading concept data: {str(e)}")
        return []

def test_concept_csv():
    """Test loading the CSV version of the concept dictionary"""
    csv_path = "data/concept_dictionary.csv"
    logger.info(f"Checking if CSV file exists: {csv_path}")
    if os.path.exists(csv_path):
        logger.info(f"CSV file found: {csv_path}")
        
        # Try to read first few lines
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            logger.info(f"CSV loaded successfully. Shape: {df.shape}")
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"CSV sample: {df.head(3)}")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
    else:
        logger.error(f"CSV file not found: {csv_path}")

def test_manifesto_concepts():
    """Test loading the manifesto concepts JSON file"""
    json_path = "data/manifesto_concepts.json"
    logger.info(f"Checking if manifesto concepts file exists: {json_path}")
    if os.path.exists(json_path):
        logger.info(f"Manifesto concepts file found: {json_path}")
        
        # Try to read the file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Manifesto concepts JSON loaded successfully.")
            if isinstance(data, dict):
                logger.info(f"Number of keys: {len(data.keys())}")
                logger.info(f"Sample keys: {list(data.keys())[:5]}")
            elif isinstance(data, list):
                logger.info(f"Number of items: {len(data)}")
                logger.info(f"Sample items: {data[:3]}")
        except Exception as e:
            logger.error(f"Error loading manifesto concepts: {str(e)}")
    else:
        logger.error(f"Manifesto concepts file not found: {json_path}")

def analyze_concept_data(concepts):
    """Analyze the loaded concept data"""
    if not concepts:
        logger.error("No concepts loaded, cannot analyze")
        return
    
    # Check the structure of the concept data
    logger.info(f"Number of concepts: {len(concepts)}")
    
    if concepts and isinstance(concepts, list):
        # Get a sample concept
        sample_concept = concepts[0]
        logger.info(f"Sample concept structure: {sample_concept.keys()}")
        
        # Get concept stats
        concept_counts = [c.get("count", 0) for c in concepts]
        if concept_counts:
            logger.info(f"Concept count stats - min: {min(concept_counts)}, max: {max(concept_counts)}, avg: {sum(concept_counts)/len(concept_counts):.2f}")
        
        # Get highest frequency concepts
        top_concepts = sorted(concepts, key=lambda x: x.get("count", 0), reverse=True)[:10]
        logger.info("Top 10 concepts by frequency:")
        for i, concept in enumerate(top_concepts, 1):
            name = concept.get("concept", "Unknown")
            count = concept.get("count", 0)
            related = concept.get("related_terms", [])
            logger.info(f"{i}. {name}: {count} occurrences, {len(related)} related terms")
    else:
        logger.error(f"Unexpected concept data format: {type(concepts)}")

def generate_fallback_json():
    """Generate a minimal concept dictionary if the real one isn't available"""
    logger.info("Generating fallback concept dictionary...")
    
    fallback_concepts = [
        {
            "concept": "bourgeoisie",
            "count": 79,
            "related_terms": ["bourgeois", "capitalists", "middle class", "property owners"]
        },
        {
            "concept": "proletariat",
            "count": 55,
            "related_terms": ["proletarian", "workers", "working class", "labor"]
        },
        {
            "concept": "communism",
            "count": 40,
            "related_terms": ["communist", "communistic", "socialist", "revolutionary"]
        },
        {
            "concept": "class struggle",
            "count": 25,
            "related_terms": ["class antagonism", "class conflict", "revolution", "oppression"]
        },
        {
            "concept": "capital",
            "count": 32,
            "related_terms": ["money", "property", "means of production", "wealth"]
        }
    ]
    
    fallback_path = "data/fallback_concepts.json"
    try:
        with open(fallback_path, 'w', encoding='utf-8') as f:
            json.dump(fallback_concepts, f, indent=2)
        logger.info(f"Fallback concept dictionary written to {fallback_path}")
    except Exception as e:
        logger.error(f"Error writing fallback concept dictionary: {str(e)}")

def main():
    """Main test function"""
    logger.info("Starting concept map data test")
    
    # First check files in the data directory
    logger.info("Checking data directory content:")
    data_files = os.listdir("data") if os.path.exists("data") else []
    logger.info(f"Files in data directory: {data_files}")
    
    # Test concept dictionary JSON loading
    concepts = load_concept_data()
    analyze_concept_data(concepts)
    
    # Test CSV version
    test_concept_csv()
    
    # Test manifesto concepts
    test_manifesto_concepts()
    
    # Generate fallback if needed
    if not concepts:
        generate_fallback_json()
        # Try loading the fallback
        fallback_concepts = load_concept_data("data/fallback_concepts.json")
        analyze_concept_data(fallback_concepts)
    
    logger.info("Concept map data test completed")

if __name__ == "__main__":
    main() 