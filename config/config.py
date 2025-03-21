"""
Central configuration settings for the Anti-Corruption RAG system.

This file serves as the single source of truth for all configurable parameters
across all components of the system.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# ============================================================
# SYSTEM PATHS AND DIRECTORIES
# ============================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Document directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
EXPORT_DIR = DATA_DIR / "exports"

# Index paths
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pickle"
VECTOR_INDEX_PATH = DATA_DIR / "vector_index"

# Create all required directories
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RAW_DATA_DIR, 
                  PROCESSED_DATA_DIR, OCR_CACHE_DIR, EXPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# DOCUMENT PROCESSING SETTINGS
# ============================================================

# OCR settings
OCR_ENABLED = True
OCR_PARALLEL_JOBS = max(1, os.cpu_count() - 2)  # Leave 2 cores free
OCR_DPI = 300
OCR_LANGUAGE = "eng"  # Tesseract language code
OCR_THRESHOLD = 200   # Image binarization threshold

# Document file extensions
SUPPORTED_DOCUMENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

# ============================================================
# CHUNKING SETTINGS
# ============================================================

# Semantic chunking
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 64  # tokens
MIN_CHUNK_SIZE = 128  # minimum size for a chunk in tokens
MAX_CHUNK_SIZE = 1024  # maximum size for a chunk in tokens
CHUNK_SIMILARITY_THRESHOLD = 0.75  # cosine similarity threshold for combining chunks

# ============================================================
# ENTITY EXTRACTION SETTINGS
# ============================================================

# Entities
ENTITY_CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence for entity extraction
ENTITY_TYPES = ["person", "organization", "location", "money", "law", "event", "norp"]

# Entity type mapping for standardization
ENTITY_TYPE_MAPPING = {
    'person': 'person',
    'per': 'person',
    'organization': 'organization',
    'org': 'organization',
    'gpe': 'location',  # Geopolitical entity
    'loc': 'location',
    'location': 'location',
    'money': 'money',
    'product': 'product',
    'law': 'law',
    'norp': 'norp',   # Nationalities, religious or political groups
    'event': 'event'
}

# Unwanted entity types to skip
UNWANTED_ENTITY_TYPES = [
    'cardinal', 'date', 'fac', 'language', 'quantity',
    'work_of_art', 'time', 'ordinal', 'percent', 'duration'
]

# Deduplication
ENTITY_DEDUPLICATION_THRESHOLD = 0.90  # Levenshtein similarity threshold
ENTITY_BATCH_SIZE = 100  # Batch size for entity processing

# Relationships
RELATIONSHIP_CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for relationships
USE_RELATIONSHIP_EXTRACTION = True  # Enable/disable relationship extraction
RELATIONSHIP_MODEL_PATH = "relations"  # Path to Flair relationship model

# ============================================================
# COREFERENCE RESOLUTION SETTINGS
# ============================================================

ENABLE_COREFERENCE = True  # Enable/disable coreference resolution
COREFERENCE_MODEL = "maverick"  # Name of the coreference model to use
COREFERENCE_BATCH_SIZE = 10  # Number of chunks to process in one batch

# ============================================================
# MODEL SETTINGS
# ============================================================

# NLP Models
NER_MODEL = "flair/ner-english-ontonotes-fast"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# LLM settings
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
MAX_HISTORY_LENGTH = 10  # Maximum number of conversation turns to keep

# External API settings
DEEPSEEK_API_ENABLED = False  # Set to True to use DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEEPSEEK_USE_REASONER = True

# ============================================================
# HARDWARE AND RESOURCE SETTINGS
# ============================================================

# GPU settings
USE_GPU = True
GPU_ID = 0  # For CUDA device selection
GPU_MEMORY_FRACTION = 0.9  # Maximum fraction of GPU memory to use

# Memory management
MAX_MEMORY_PERCENTAGE = 80.0  # Unload models at this memory threshold
MAX_GPU_MEMORY_PERCENTAGE = 80.0  # Unload models at this GPU memory threshold
RESOURCE_CHECK_INTERVAL = 60  # Check resource usage every X seconds

# Model sizes for resource management (in GB)
MODEL_SIZES = {
    "chunker_embedding": 1.2,
    "coreference_model": 1.5,
    "ner_model": 1.0,
    "relation_model": 1.0,
    "embedding_model": 1.2,
    "reranker_model": 3.0,
    "llm_model": 6.0
}

# Model priorities (higher value = higher priority to keep in memory)
MODEL_PRIORITIES = {
    "chunker_embedding": 3,
    "coreference_model": 2,
    "ner_model": 2,
    "relation_model": 2,
    "embedding_model": 4,
    "reranker_model": 4,
    "llm_model": 5
}

# ============================================================
# VECTOR DATABASE SETTINGS
# ============================================================

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "ti_rag_collection"
QDRANT_DISTANCE = "Cosine"
QDRANT_VECTOR_SIZE = EMBEDDING_DIMENSION

# ============================================================
# SEARCH AND QUERY SETTINGS
# ============================================================

# Hybrid search
BM25_WEIGHT = 0.5  # Weight for BM25 scores in hybrid search
VECTOR_WEIGHT = 0.5  # Weight for vector scores in hybrid search
BM25_K1 = 60.0  # BM25 parameter k1
BM25_B = 0.75  # BM25 parameter b

# Retrieval settings
RETRIEVE_TOP_K = 5  # Number of chunks to retrieve
RERANK_TOP_K = 3  # Number of chunks after reranking
MINIMUM_SCORE_THRESHOLD = 0.6  # Minimum score to include a chunk

# Query system
CONVERSATION_EXPIRY = 86400  # Seconds until conversation context expires (24 hours)
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant for anti-corruption research. 
Use the provided document chunks to answer questions accurately.
If the answer is not in the documents, say so clearly."""

# ============================================================
# UI SETTINGS
# ============================================================

# Streamlit theme
STREAMLIT_THEME = {
    "primaryColor": "#1e3a8a",
    "backgroundColor": "#f8f9fa",
    "secondaryBackgroundColor": "#ffffff",
    "textColor": "#1e293b",
    "font": "Inter"
}

# Entity visualization colors
ENTITY_COLORS = {
    'person': '#3b82f6',       # Blue
    'organization': '#10b981', # Green
    'location': '#f59e0b',     # Amber
    'money': '#8b5cf6',        # Purple
    'law': '#ec4899',          # Pink
    'event': '#ef4444',        # Red
    'norp': '#0d9488',         # Teal
    'default': '#94a3b8'       # Gray
}

# Network visualization
NETWORK_HEIGHT = 600  # Height of network visualization in pixels
NETWORK_WIDTH = "100%"  # Width of network visualization
NETWORK_BGCOLOR = "#ffffff"  # Background color
NETWORK_FONT_COLOR = "#333333"  # Font color

# ============================================================
# EXHAUSTIVE EXTRACTION SETTINGS
# ============================================================

# Pydantic model extraction defaults
DEFAULT_EXTRACTION_TEMPERATURE = 0.1
EXTRACTION_MAX_TOKENS = 2000
MAX_EXTRACTION_BATCH_SIZE = 20  # Maximum number of chunks to process in one batch
