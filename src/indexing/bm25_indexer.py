"""
BM25 indexing module for keyword-based retrieval.
"""
from typing import List, Dict, Any, Union, Set, Tuple
import os
import json
import pickle
import re
import nltk
from rank_bm25 import BM25Okapi
import numpy as np

from config.config import BM25_INDEX_PATH
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "bm25_indexer.log")

class BM25Indexer:
    """
    BM25 indexer for keyword-based search.
    """
    
    def __init__(self, index_path=None):
        """
        Initialize BM25 indexer.
        
        Args:
            index_path: Path to save/load the index
        """
        self.index_path = index_path or BM25_INDEX_PATH
        self.tokenized_corpus = []
        self.chunks = []
        self.bm25 = None
        self.initialized = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Try to load existing index
        self._load_index()
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        logger.info(f"Initialized BM25Indexer with index_path={self.index_path}")
    
    def _load_index(self):
        """
        Load BM25 index from disk if it exists.
        """
        if os.path.exists(f"{self.index_path}.pickle"):
            try:
                logger.info("Loading BM25 index from disk")
                with open(f"{self.index_path}.pickle", 'rb') as f:
                    data = pickle.load(f)
                    
                    self.bm25 = data['bm25']
                    self.tokenized_corpus = data['tokenized_corpus']
                    self.chunks = data['chunks']
                    self.initialized = True
                    
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.initialized = False
    
    def _save_index(self):
        """
        Save BM25 index to disk.
        """
        try:
            logger.info("Saving BM25 index to disk")
            with open(f"{self.index_path}.pickle", 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'tokenized_corpus': self.tokenized_corpus,
                    'chunks': self.chunks
                }, f)
            logger.info(f"Saved BM25 index with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for indexing or searching.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove entity markers if present
        text = re.sub(r'##entity:[a-z]+##', '', text)
        text = re.sub(r'##/[a-z]+##', '', text)
        
        # Replace non-alphanumeric with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Preprocess
        text = self._preprocess_text(text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove very short tokens (length 1 or 2)
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks to the BM25 index.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding {len(chunks)} chunks to BM25 index")
        
        # Tokenize new chunks
        new_tokenized_chunks = []
        for chunk in chunks:
            # Get the text (use text_with_entities if available, otherwise use text)
            text = chunk.get('text_with_entities', chunk.get('text', ''))
            tokens = self._tokenize(text)
            new_tokenized_chunks.append(tokens)
        
        # Add to existing corpus
        self.tokenized_corpus.extend(new_tokenized_chunks)
        self.chunks.extend(chunks)
        
        # Create or update BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.initialized = True
        
        # Save index
        self._save_index()
        
        logger.info(f"Added {len(chunks)} chunks to BM25 index, total: {len(self.chunks)}")
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching chunks with scores
        """
        if not self.initialized or len(self.chunks) == 0:
            logger.warning("BM25 index not initialized or empty")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Query tokens are empty after preprocessing")
            return []
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            # Skip if score is too low
            if scores[idx] <= 0:
                continue
            
            chunk = self.chunks[idx]
            result = {
                'chunk_id': chunk.get('chunk_id', ''),
                'document_id': chunk.get('document_id', ''),
                'file_name': chunk.get('file_name', ''),
                'text': chunk.get('text', ''),
                'score': float(scores[idx]),
                'page_num': chunk.get('page_num', None),
                'metadata': chunk.get('metadata', {})
            }
            results.append(result)
        
        logger.info(f"BM25 search for '{query}' returned {len(results)} results")
        return results
    
    def clear_index(self) -> bool:
        """
        Clear the BM25 index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Clearing BM25 index")
            self.tokenized_corpus = []
            self.chunks = []
            self.bm25 = None
            self.initialized = False
            
            # Remove files if they exist
            if os.path.exists(f"{self.index_path}.pickle"):
                os.remove(f"{self.index_path}.pickle")
            
            logger.info("BM25 index cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear BM25 index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'initialized': self.initialized,
            'num_chunks': len(self.chunks),
            'num_documents': len(set(chunk.get('document_id', '') for chunk in self.chunks)),
            'num_tokens': sum(len(tokens) for tokens in self.tokenized_corpus) if self.tokenized_corpus else 0,
            'avg_tokens_per_chunk': np.mean([len(tokens) for tokens in self.tokenized_corpus]) if self.tokenized_corpus else 0,
            'index_file_size_bytes': os.path.getsize(f"{self.index_path}.pickle") if os.path.exists(f"{self.index_path}.pickle") else 0
        }
        
        return stats