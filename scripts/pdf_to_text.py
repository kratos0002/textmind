#!/usr/bin/env python3
import os
import argparse
import re
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Install with 'pip install pymupdf'")
    exit(1)

def extract_text_from_pdf(pdf_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract text from a PDF file, preserving paragraphs and sections.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text (if None, won't save to file)
        
    Returns:
        Extracted text as a string
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    all_text = []
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Clean up text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines with double newlines
        all_text.append(text)
    
    # Join all pages with double newlines
    full_text = '\n\n'.join(all_text)
    
    # Further text cleanup for section titles
    # Look for potential section titles (uppercase text followed by newlines)
    full_text = re.sub(r'([A-Z][A-Z\s]+[A-Z])\n', r'\n\n\1\n\n', full_text)
    
    # Additional cleanup - remove extra whitespace
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    # Save to file if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Text saved to {output_path}")
    
    return full_text

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to Text')
    parser.add_argument('input_pdf', help='Path to the input PDF file')
    parser.add_argument('--output', '-o', help='Path to save the output text file (default: same name with .txt extension)')
    
    args = parser.parse_args()
    
    input_pdf = args.input_pdf
    output_file = args.output
    
    if not output_file:
        # Replace .pdf extension with .txt
        output_file = os.path.splitext(input_pdf)[0] + '.txt'
    
    try:
        extract_text_from_pdf(input_pdf, output_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 