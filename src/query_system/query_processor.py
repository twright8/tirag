"""
Query processor module for handling conversational queries with VLLM.
"""
from typing import List, Dict, Any, Optional, Tuple
import time
import gc
import torch
import json
from datetime import datetime

from src.query_system.hybrid_searcher import HybridSearcher
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage, free_memory

logger = setup_logger(__name__, "query_processor.log")

class QueryProcessor:
    """
    Query processor for handling conversational queries with VLLM.
    """
    
    def __init__(self, hybrid_searcher: Optional[HybridSearcher] = None):
        """
        Initialize query processor.
        
        Args:
            hybrid_searcher: Hybrid searcher for retrieving relevant chunks
        """
        self.hybrid_searcher = hybrid_searcher or HybridSearcher()
        self.llm = None
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.model_loaded = False
        self.conversation_history = []
        
        logger.info(f"Initialized QueryProcessor with model={self.model_name}")
        log_memory_usage(logger)
    
    def load_model(self):
        """
        Load the VLLM model.
        """
        if self.model_loaded:
            return
        
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading VLLM model: {self.model_name}")
            
            # Load model
            self.llm = LLM(
                self.model_name,
                gpu_memory_utilization=1.0,
                dtype="half",
                quantization="fp8",
                enable_chunked_prefill=False,
                enforce_eager=True,
                enable_prefix_caching=True,
                trust_remote_code=True,
                tensor_parallel_size=1,
            )
            
            self.model_loaded = True
            logger.info(f"VLLM model loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading VLLM model: {e}")
            raise
    
    def unload_model(self):
        """
        Unload the VLLM model to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info(f"Unloading VLLM model")
            
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
            
            logger.info(f"VLLM model unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading VLLM model: {e}")
    
    def process_query(self, query: str, top_k: int = 5, 
                     filter_conditions: Optional[Dict] = None,
                     conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query using RAG.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_conditions: Optional filter conditions for search
            conversation_id: Optional conversation ID for context
            
        Returns:
            Response with answer and context
        """
        start_time = time.time()
        
        if not query:
            return {
                'answer': "I don't understand your question. Could you please rephrase it?",
                'context': [],
                'execution_time': 0.0
            }
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        search_results, search_time = self.hybrid_searcher.search(
            query=query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            rerank=True
        )
        
        # Generate answer
        answer, generation_time = self._generate_answer(query, search_results, conversation_id)
        
        total_time = time.time() - start_time
        
        # Prepare response
        response = {
            'answer': answer,
            'context': search_results,
            'search_time': search_time,
            'generation_time': generation_time,
            'execution_time': total_time
        }
        
        logger.info(f"Query processed in {total_time:.2f}s "
                   f"(search: {search_time:.2f}s, generation: {generation_time:.2f}s)")
        
        return response
    
    def _generate_answer(self, query: str, search_results: List[Dict[str, Any]], 
                        conversation_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate an answer using the language model.
        
        Args:
            query: User query
            search_results: Retrieved chunks
            conversation_id: Optional conversation ID for context
            
        Returns:
            Tuple of (answer, execution time)
        """
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not self.model_loaded:
                self.load_model()
            
            # Prepare context from search results
            context = self._prepare_context(search_results)
            
            # Prepare prompt
            system_prompt = """You are a helpful anti-corruption analysis assistant. 
You help analyze documents related to corruption, financial crimes, and related investigations.
Answer the user's question based only on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Do not make up information that is not in the context.
Cite the document and page number where you found the information when possible."""
            
            user_prompt = f"""Question: {query}

Context:
{context}

Please answer the question based on the context above."""
            
            # Format prompt for VLLM
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate answer
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9,
            )
            
            outputs = self.llm.generate(formatted_prompt, sampling_params)
            
            # Extract answer from response
            answer = outputs[0].outputs[0].text.strip()
            
            execution_time = time.time() - start_time
            
            logger.info(f"Generated answer in {execution_time:.2f}s")
            
            return answer, execution_time
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, but I encountered an error while processing your question. Please try again.", time.time() - start_time
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Prepare context from search results.
        
        Args:
            search_results: Retrieved chunks
            
        Returns:
            Formatted context
        """
        if not search_results:
            return "No relevant information found."
        
        # Format context
        context_parts = []
        
        for i, result in enumerate(search_results):
            doc_id = result.get('document_id', 'Unknown')
            file_name = result.get('file_name', 'Unknown')
            page_num = result.get('page_num', None)
            text = result.get('text', '')
            score = result.get('score', 0.0)
            
            # Format document information
            doc_info = f"[Document: {file_name}"
            if page_num:
                doc_info += f", Page: {page_num}"
            doc_info += f", Score: {score:.2f}]"
            
            # Add to context
            context_parts.append(f"CHUNK {i+1}:\n{doc_info}\n{text.strip()}\n")
        
        return "\n\n".join(context_parts)
    
    def clear_conversation_history(self, conversation_id: Optional[str] = None):
        """
        Clear conversation history.
        
        Args:
            conversation_id: Optional conversation ID to clear
        """
        if conversation_id:
            # Clear specific conversation
            self.conversation_history = [
                conv for conv in self.conversation_history 
                if conv.get('conversation_id') != conversation_id
            ]
        else:
            # Clear all conversations
            self.conversation_history = []
        
        logger.info(f"Cleared conversation history")
    
    def get_conversation_history(self, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Optional conversation ID to get
            
        Returns:
            List of conversation messages
        """
        if conversation_id:
            # Get specific conversation
            return [
                conv for conv in self.conversation_history 
                if conv.get('conversation_id') == conversation_id
            ]
        else:
            # Get all conversations
            return self.conversation_history