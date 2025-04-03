#!/usr/bin/env python3
import os
import argparse
import sys

try:
    from booknlp.booknlp import BookNLP
except ImportError:
    print("BookNLP not installed. Install with 'pip install booknlp'")
    sys.exit(1)

def process_text_with_booknlp(input_file, output_dir, book_id="manifesto", overwrite=True):
    """
    Process a text file with BookNLP.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save BookNLP outputs
        book_id: ID for the book (used in output filenames)
        overwrite: Whether to overwrite existing files
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize BookNLP with English language model - updated based on documentation
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "small"  # Use small model for faster processing
    }
    booknlp = BookNLP("en", model_params)
    
    # Process the text file - update parameter names to match API
    print(f"Processing {input_file} with BookNLP...")
    booknlp.process(
        input_file,
        output_dir,
        book_id
    )
    
    print(f"BookNLP processing complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process text with BookNLP')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output-dir', '-o', default='output/booknlp',
                        help='Directory to save BookNLP outputs (default: output/booknlp)')
    parser.add_argument('--book-id', '-b', default='manifesto',
                        help='ID for the book (used in output filenames) (default: manifesto)')
    parser.add_argument('--no-overwrite', action='store_false', dest='overwrite',
                        help='Do not overwrite existing files')
    
    args = parser.parse_args()
    
    try:
        process_text_with_booknlp(
            args.input_file,
            args.output_dir,
            args.book_id,
            args.overwrite
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 