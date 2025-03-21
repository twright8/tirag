"""
Setup script to initialize the Anti-Corruption RAG system.
This creates necessary directories and downloads required models.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
from config.config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    OCR_CACHE_DIR, MODEL_DIR, BM25_INDEX_PATH
)

def create_directories():
    """Create necessary directories for the system."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OCR_CACHE_DIR,
        MODEL_DIR,
        os.path.dirname(BM25_INDEX_PATH)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_required_models():
    """Download required models for the system."""
    try:
        # Try importing the required libraries
        import nltk
        import flair
        import sentence_transformers
        
        # Download NLTK resources
        nltk.download('punkt')
        print("Downloaded NLTK punkt tokenizer")
        
        # Download Flair NER model
        from flair.nn import Classifier
        print("Checking for Flair NER model...")
        _ = Classifier.load('flair/ner-english-ontonotes-fast')
        print("Flair NER model available")
        
        # Download sentence transformers model
        from sentence_transformers import SentenceTransformer
        print("Checking for embedding model...")
        _ = SentenceTransformer('intfloat/multilingual-e5-base')
        print("Embedding model available")
        
        print("All required models are downloaded and available")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        print("Please install the required packages and run this script again.")
        sys.exit(1)

def setup():
    """Set up the Anti-Corruption RAG system."""
    print("Setting up Anti-Corruption RAG system...")
    
    # Create directories
    create_directories()
    
    # Download required models
    download_required_models()
    
    print("Setup complete!")

if __name__ == "__main__":
    setup()