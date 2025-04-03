# TextMind: Semantic Q&A for The Communist Manifesto

TextMind is a natural language question-answering system specifically designed for The Communist Manifesto. It leverages NLP techniques to extract structured data from the text, categorize questions, and provide source-cited answers to user queries.

## Project Overview

TextMind creates a semantic interface between users and Marx's seminal text, allowing for:
- Definition queries (e.g., "What is the bourgeoisie?")
- Contextual questions (e.g., "When was the Manifesto written?")
- Argument analysis (e.g., "Why does Marx believe capitalism will collapse?")
- Evidence examination (e.g., "What examples does Marx provide for worker exploitation?")
- Analogical reasoning (e.g., "How does Marx compare capitalism to feudalism?")

## Repository Structure

```
├── data/                     # Input text files and processed data
│   ├── manifesto.pdf         # Original PDF document
│   ├── manifesto.txt         # Extracted plain text
│   ├── manifesto_clean.jsonl # Structured data with metadata
│   ├── manifesto_sections.json # Section information
│   ├── manifesto_concepts.json # Extracted theoretical concepts
│   ├── concept_dictionary.json # Dictionary of concept relationships
│   └── entity_dictionary.csv # Dictionary of entities in the text
│
├── output/                   # Generated outputs
│   ├── booknlp/              # BookNLP processing results
│   │   ├── manifesto.tokens  # Token-level annotations
│   │   └── manifesto.supersense # Semantic categories
│   └── classification_tests.json # Question classification test results
│
├── scripts/                  # Processing scripts
│   ├── pdf_to_text.py        # PDF to text converter
│   ├── run_booknlp.py        # BookNLP processor 
│   ├── extract_from_tokens.py # Extract structured data
│   ├── concept_tagging.py    # Concept tagging system
│   └── question_classifier.py # Question intent classifier
│
└── notebooks/                # Jupyter notebooks
    └── postprocess.ipynb     # Data post-processing
```

## Features Implemented

### 1. Data Extraction Layer
- PDF to text conversion preserving paragraphs and sections
- BookNLP processing for entity recognition and semantic tagging
- Structured output with section and paragraph metadata

### 2. Concept Analysis
- Extraction of theoretical concepts (e.g., "class struggle", "bourgeoisie", "proletariat")
- Concept dictionary with related terms and contexts
- Supersense tagging for semantic categories

### 3. Question Understanding
- Zero-shot classification of question intent
- Hybrid model combining NLP and rule-based patterns
- Support for 5 question categories (Definition, Context, Argument, Evidence, Analogy)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kratos0002/textmind.git
cd textmind
```

2. Install dependencies:
```bash
pip install pymupdf booknlp pandas numpy spacy transformers torch jupyter
python -m spacy download en_core_web_sm
```

## Usage

### Processing the Text
```bash
# Convert PDF to text
python scripts/pdf_to_text.py data/manifesto.pdf

# Process with BookNLP
python scripts/run_booknlp.py data/manifesto.txt

# Extract structured data
python scripts/extract_from_tokens.py

# Tag theoretical concepts
python scripts/concept_tagging.py data/manifesto_clean.jsonl
```

### Classifying Questions
```bash
# Classify a single question
python scripts/question_classifier.py --question "What is the bourgeoisie according to Marx?"

# Run test suite
python scripts/question_classifier.py --test
```

## Future Development
- Vector embedding for semantic search
- Knowledge graph creation
- Full Q&A pipeline with source citations
- Web interface for interactive exploration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 