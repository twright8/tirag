"""
Hybrid search module combining BM25 and vector search with reciprocal rank fusion.
"""
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np

from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embedding_indexer import EmbeddingIndexer
from src.utils.logger import setup_logger, log_query_result

logger = setup_logger(__name__, "hybrid_searcher.log")

class HybridSearcher:
    """
    Hybrid search module combining BM25 and vector search with reciprocal rank fusion.
    """
    
    def __init__(self, bm25_indexer: Optional[BM25Indexer] = None, 
                embedding_indexer: Optional[EmbeddingIndexer] = None,
                k1: float = 60.0, 
                bm25_weight: float = 0.5):
        """
        Initialize hybrid searcher.
        
        Args:
            bm25_indexer: BM25 indexer
            embedding_indexer: Embedding indexer
            k1: Constant for RRF score (controls how quickly ranks lose value)
            bm25_weight: Weight for BM25 vs embedding search (0-1)
        """
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.embedding_indexer = embedding_indexer or EmbeddingIndexer()
        self.k1 = k1
        self.bm25_weight = bm25_weight
        
        logger.info(f"Initialized HybridSearcher with k1={k1}, bm25_weight={bm25_weight}")
    
    def search(self, query: str, top_k: int = 5, 
              filter_conditions: Optional[Dict] = None,
              rerank: bool = True,
              use_bm25: bool = True,
              use_embeddings: bool = True) -> Tuple[List[Dict[str, Any]], float]:
        """
        Perform hybrid search using reciprocal rank fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            rerank: Whether to rerank results
            use_bm25: Whether to use BM25 search
            use_embeddings: Whether to use embedding search
            
        Returns:
            Tuple of (search results, execution time)
        """
        if not query:
            return [], 0.0
        
        start_time = time.time()
        
        logger.info(f"Hybrid search for: {query}")
        
        # Determine which search methods to use
        if not use_bm25 and not use_embeddings:
            logger.warning("Both search methods disabled, defaulting to embedding search")
            use_embeddings = True
        
        # Get more results than needed for fusion
        search_k = max(top_k * 3, 20)
        
        # Run BM25 search
        bm25_results = []
        if use_bm25:
            bm25_results = self.bm25_indexer.search(query, search_k)
        
        # Run embedding search
        embedding_results = []
        if use_embeddings:
            embedding_results = self.embedding_indexer.search(query, search_k, filter_conditions)
        
        # Combine results with reciprocal rank fusion
        if use_bm25 and use_embeddings:
            results = self._reciprocal_rank_fusion(bm25_results, embedding_results, top_k)
        elif use_bm25:
            results = bm25_results[:top_k]
        else:
            results = embedding_results[:top_k]
        
        # Rerank if requested
        if rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        execution_time = time.time() - start_time
        
        # Log query results
        log_query_result(logger, query, results, execution_time)
        
        return results, execution_time
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Dict[str, Any]], 
                              embedding_results: List[Dict[str, Any]], 
                              top_k: int) -> List[Dict[str, Any]]:
        """
        Combine BM25 and embedding search results using reciprocal rank fusion.
        
        Args:
            bm25_results: BM25 search results
            embedding_results: Embedding search results
            top_k: Number of results to return
            
        Returns:
            Combined search results
        """
        logger.info("Performing reciprocal rank fusion")
        
        # Create mappings from chunk_id to rank
        bm25_ranks = {result['chunk_id']: i+1 for i, result in enumerate(bm25_results)}
        embedding_ranks = {result['chunk_id']: i+1 for i, result in enumerate(embedding_results)}
        
        # Create a set of all unique chunk_ids
        all_chunk_ids = set(bm25_ranks.keys()) | set(embedding_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            # Get ranks (default to a large value if not found)
            bm25_rank = bm25_ranks.get(chunk_id, len(bm25_ranks) + 100)
            embedding_rank = embedding_ranks.get(chunk_id, len(embedding_ranks) + 100)
            
            # Calculate RRF score with weight
            bm25_score = 1.0 / (self.k1 + bm25_rank)
            embedding_score = 1.0 / (self.k1 + embedding_rank)
            
            # Apply weights
            rrf_scores[chunk_id] = (bm25_score * self.bm25_weight + 
                                   embedding_score * (1 - self.bm25_weight))
        
        # Sort chunk_ids by RRF score (descending)
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        # Create result list
        results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            # Find original result in either list
            result = None
            for r in bm25_results:
                if r['chunk_id'] == chunk_id:
                    result = r.copy()
                    break
            
            if result is None:
                for r in embedding_results:
                    if r['chunk_id'] == chunk_id:
                        result = r.copy()
                        break
            
            if result:
                # Add RRF score
                result['rrf_score'] = rrf_scores[chunk_id]
                results.append(result)
        
        logger.info(f"Reciprocal rank fusion returned {len(results)} results")
        return results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a reranking model (VLLM).
        
        Args:
            query: Search query
            results: Search results to rerank
            
        Returns:
            Reranked search results
        """
        try:
            logger.info(f"Reranking {len(results)} results")
            
            # Import VLLM for reranking
            from vllm import LLM
            
            # Load reranker model
            reranker = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
            
            # Prepare query-document pairs
            texts = [result['text'] for result in results]
            
            # Score each pair
            reranked_scores = []
            for text in texts:
                (output,) = reranker.score(query, text)
                score = output.outputs.score
                reranked_scores.append(score)
            
            # Update scores and sort by new scores
            for i, result in enumerate(results):
                result['original_score'] = result.get('score', 0.0)
                result['score'] = reranked_scores[i]
            
            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Clean up reranker
            del reranker
            
            logger.info(f"Reranking complete")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results  # Return original results on error
    
    def calculate_relevance(self, query: str, text: str) -> float:
        """
        Calculate relevance score between query and text.
        
        Args:
            query: Search query
            text: Text to calculate relevance for
            
        Returns:
            Relevance score (0-1)
        """
        try:
            # Load reranker model
            from vllm import LLM
            reranker = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
            
            # Score query-document pair
            (output,) = reranker.score(query, text)
            score = output.outputs.score
            
            # Clean up reranker
            del reranker
            
            return score
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0