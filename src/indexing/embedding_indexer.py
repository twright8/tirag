"""
Embedding indexer module for semantic retrieval using VLLM embeddings.
"""
from typing import List, Dict, Any, Union, Optional
import os
import gc
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid
import time

from config.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage, free_memory

logger = setup_logger(__name__, "embedding_indexer.log")

class EmbeddingIndexer:
    """
    Embedding indexer using VLLM and Qdrant vector database.
    """
    
    def __init__(self, model_name: str = None, 
                collection_name: str = None,
                qdrant_host: str = None,
                qdrant_port: int = None):
        """
        Initialize embedding indexer.
        
        Args:
            model_name: Embedding model name
            collection_name: Qdrant collection name
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.qdrant_host = qdrant_host or QDRANT_HOST
        self.qdrant_port = qdrant_port or QDRANT_PORT
        self.embedding_dim = EMBEDDING_DIMENSION
        
        # Initialize model
        self.llm = None
        self.model_loaded = False
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        self._ensure_collection_exists()
        
        logger.info(f"Initialized EmbeddingIndexer with model={self.model_name}, "
                   f"collection={self.collection_name}, qdrant={self.qdrant_host}:{self.qdrant_port}")
        
        log_memory_usage(logger)
    
    def _ensure_collection_exists(self):
        """
        Ensure Qdrant collection exists, create it if it doesn't.
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_dim,
                        distance=qmodels.Distance.COSINE
                    )
                )
                
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {e}")
            raise
    
    def load_model(self):
        """
        Load the VLLM embedding model.
        """
        if self.model_loaded:
            return
        
        try:
            from vllm import LLM
            
            logger.info(f"Loading VLLM embedding model: {self.model_name}")
            
            # Load model with embedding task
            self.llm = LLM(model=self.model_name, task="embed")
            
            self.model_loaded = True
            logger.info(f"VLLM embedding model loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading VLLM embedding model: {e}")
            raise
    
    def unload_model(self):
        """
        Unload the VLLM model to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info(f"Unloading VLLM embedding model")
            
            # Clean up VLLM resources
            from vllm.distributed.parallel_state import destroy_model_parallel
            
            # Delete the llm object and free the memory
            destroy_model_parallel()
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'driver_worker'):
                del self.llm.llm_engine.driver_worker
            del self.llm
            self.llm = None
            self.model_loaded = False
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            
            logger.info(f"VLLM embedding model unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading VLLM embedding model: {e}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                # Generate embeddings
                outputs = self.llm.embed(batch)
                
                # Extract embeddings from outputs
                batch_embeddings = [output.outputs.embedding for output in outputs]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Return empty embeddings in case of error
                return []
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add chunks to the vector database.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            return True
        
        logger.info(f"Adding {len(chunks)} chunks to vector database")
        
        try:
            # Extract texts for embedding
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings or len(embeddings) != len(chunks):
                logger.error(f"Failed to generate embeddings for chunks")
                return False
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create a unique ID for the point
                point_id = str(uuid.uuid4())
                
                # Create payload
                payload = {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'document_id': chunk.get('document_id', ''),
                    'file_name': chunk.get('file_name', ''),
                    'text': chunk.get('text', ''),
                    'page_num': chunk.get('page_num', None),
                    'chunk_idx': chunk.get('chunk_idx', None),
                    'metadata': chunk.get('metadata', {})
                }
                
                # Create point
                point = qmodels.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                
                points.append(point)
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} points to Qdrant collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of matching chunks with scores
        """
        if not query:
            return []
        
        logger.info(f"Searching for: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Convert filter conditions to Qdrant filter if provided
            qdrant_filter = None
            if filter_conditions:
                qdrant_filter = self._create_qdrant_filter(filter_conditions)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            # Process results
            results = []
            for result in search_results:
                # Get payload
                payload = result.payload
                
                # Create result dict
                result_dict = {
                    'chunk_id': payload.get('chunk_id', ''),
                    'document_id': payload.get('document_id', ''),
                    'file_name': payload.get('file_name', ''),
                    'text': payload.get('text', ''),
                    'score': result.score,
                    'page_num': payload.get('page_num', None),
                    'metadata': payload.get('metadata', {})
                }
                
                results.append(result_dict)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in vector database: {e}")
            return []
    
    def _create_qdrant_filter(self, filter_conditions: Dict) -> qmodels.Filter:
        """
        Create a Qdrant filter from filter conditions.
        
        Args:
            filter_conditions: Filter conditions
            
        Returns:
            Qdrant filter
        """
        if not filter_conditions:
            return None
        
        # Handle document_id filter
        if 'document_id' in filter_conditions:
            return qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=filter_conditions['document_id'])
                    )
                ]
            )
        
        # More complex filters can be added here
        
        return None
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Clearing Qdrant collection: {self.collection_name}")
            
            # Delete the collection
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            
            # Recreate the collection
            self._ensure_collection_exists()
            
            logger.info(f"Cleared Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary of statistics
        """
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            
            # Get vector count
            collection_count = self.qdrant_client.count(collection_name=self.collection_name)
            
            stats = {
                'collection_name': self.collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance.name,
                'vector_count': collection_count.count
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting vector database stats: {e}")
            return {'error': str(e)}