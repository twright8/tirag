"""
Main entry point for the Anti-Corruption RAG application.
"""
import streamlit as st
import sys
import os
from pathlib import Path
import nltk
nltk.download('punkt_tab')
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import the Streamlit app
from src.ui.app import main

if __name__ == "__main__":
    # Configure project root
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parent)

    # Run the Streamlit app
    main()