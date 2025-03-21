"""
Document chunking module that implements semantic chunking.
"""
import uuid
import re
from typing import List, Dict, Any, Optional
import torch

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage, free_memory

logger = setup_logger(__name__, "document_chunker.log")

class DocumentChunker:
    """
    Document chunker that implements semantic chunking.
    """
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 embedding_model_name: str = None):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Size of chunks. Defaults to config value.
            chunk_overlap: Overlap between chunks. Defaults to config value.
            embedding_model_name: Embedding model name. Defaults to config value.
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL_NAME
        
        logger.info(f"Initializing DocumentChunker with chunk_size={self.chunk_size}, "
                   f"chunk_overlap={self.chunk_overlap}, embedding_model={self.embedding_model_name}")
        
        # Initialize embedding model
        self.embeddings = None
        self.model_loaded = False
        
        log_memory_usage(logger)
    
    def load_embedding_model(self):
        """
        Load the embedding model.
        """
        if self.model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = SentenceTransformer(self.embedding_model_name)
            self.model_loaded = True
            logger.info(f"Embedding model loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def unload_embedding_model(self):
        """
        Unload the embedding model to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info(f"Unloading embedding model: {self.embedding_model_name}")
            del self.embeddings
            self.embeddings = None
            self.model_loaded = False
            
            # Force garbage collection
            free_memory(logger)
            
            logger.info(f"Embedding model unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading embedding model: {e}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using semantic chunking.
        
        Args:
            document: Document data from DocumentLoader
            
        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Chunking document: {document.get('document_id', 'unknown')}")
        
        # Ensure embedding model is loaded
        if not self.model_loaded:
            self.load_embedding_model()
        
        chunks = []
        
        # Process each content item (page or section)
        for content_item in document.get('content', []):
            page_num = content_item.get('page_num', None)
            text = content_item.get('text', '')
            
            # Skip empty content
            if not text.strip():
                continue
            
            # Generate semantic chunks for this content
            content_chunks = self._semantic_chunking(text)
            
            # Create chunk objects with metadata
            for chunk_idx, chunk_text in enumerate(content_chunks):
                chunk_id = str(uuid.uuid4())
                
                chunk = {
                    'chunk_id': chunk_id,
                    'document_id': document.get('document_id', ''),
                    'file_name': document.get('file_name', ''),
                    'text': chunk_text,
                    'page_num': page_num,
                    'chunk_idx': chunk_idx,
                    'metadata': {
                        'document_metadata': document.get('metadata', {}),
                        'file_type': document.get('file_type', ''),
                        'chunk_method': 'semantic'
                    }
                }
                
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {document.get('document_id', 'unknown')}")
        log_memory_usage(logger)
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Perform semantic chunking on text using a hierarchical approach:
        1. First split by recursive text boundaries (paragraphs, sentences)
        2. Then apply semantic chunking with LangChain for more intelligent breaks
        3. Fall back to token-based splitting for oversized chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        try:
            # Step 1: First split by natural boundaries
            initial_chunks = self._split_by_recursive_boundaries(text)
            logger.info(f"Initial boundary splitting created {len(initial_chunks)} chunks")
            
            # Step 2: Apply LangChain's semantic chunking to each large chunk
            try:
                # Try to import LangChain components
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_core.documents import Document
                
                semantic_chunks = []
                
                for chunk in initial_chunks:
                    # Only apply semantic chunking to larger chunks
                    if len(chunk) > self.chunk_size:
                        # Create Document object
                        doc = Document(page_content=chunk)
                        
                        # Configure semantic splitter
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
                            is_separator_regex=False
                        )
                        
                        # Split the document
                        splits = text_splitter.split_documents([doc])
                        semantic_chunks.extend([doc.page_content for doc in splits])
                    else:
                        # Keep small chunks as-is
                        semantic_chunks.append(chunk)
                        
                logger.info(f"LangChain semantic chunking created {len(semantic_chunks)} chunks")
                
            except ImportError:
                logger.warning("LangChain not available, falling back to custom semantic chunking")
                
                semantic_chunks = []
                for chunk in initial_chunks:
                    # Only apply semantic chunking to larger chunks
                    if len(chunk) > self.chunk_size:
                        # Apply sentence-transformer based similarity chunking
                        sentences = self._split_into_sentences(chunk)
                        
                        if len(sentences) > 1:
                            # Embed sentences
                            embeddings = self.embeddings.encode(sentences)
                            
                            # Identify breakpoints based on cosine similarity
                            breakpoints = self._find_semantic_breakpoints(embeddings)
                            
                            # Create chunks based on the semantic breakpoints
                            current_chunk = []
                            for i, sentence in enumerate(sentences):
                                current_chunk.append(sentence)
                                
                                # Check if this is a breakpoint or last sentence
                                if i in breakpoints or i == len(sentences) - 1:
                                    if current_chunk:
                                        semantic_chunks.append(" ".join(current_chunk))
                                        current_chunk = []
                        else:
                            # If only one sentence, just add it
                            semantic_chunks.append(chunk)
                    else:
                        # Keep small chunks as-is
                        semantic_chunks.append(chunk)
            
            # Step 3: Apply fallback splitting for any chunks still too large
            final_chunks = []
            for chunk in semantic_chunks:
                if len(chunk) > self.chunk_size * 1.5:  # Allow some flexibility
                    # Split oversized chunks using sentence boundaries
                    final_chunks.extend(self._sentence_splitting(chunk))
                else:
                    final_chunks.append(chunk)
            
            logger.info(f"Semantic chunking pipeline created {len(final_chunks)} chunks from text of length {len(text)}")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            logger.warning("Falling back to basic chunking")
            return self._basic_chunking(text)

    def _find_semantic_breakpoints(self, embeddings) -> List[int]:
        """
        Find semantic breakpoints based on embeddings.
        
        Args:
            embeddings: List of sentence embeddings
            
        Returns:
            List of indices where semantic breaks should occur
        """
        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[i + 1]).unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        # Find potential breakpoints where similarity is lower
        breakpoints = []
        
        if not similarities:
            return breakpoints
        
        # Calculate threshold as percentile of similarities
        mean_similarity = sum(similarities) / len(similarities)
        similarity_threshold = mean_similarity * 0.8  # 80% of mean
        
        # Find valleys in the similarity curve
        for i in range(len(similarities)):
            if similarities[i] < similarity_threshold:
                # Make sure we don't have breakpoints too close together
                if not breakpoints or i - breakpoints[-1] >= 3:  # At least 3 sentences apart
                    breakpoints.append(i)
        
        # Ensure chunks are not too large
        max_chunk_size = 10  # Max number of sentences per chunk
        if not breakpoints:
            # If no natural breakpoints, create them at fixed intervals
            for i in range(max_chunk_size, len(embeddings), max_chunk_size):
                breakpoints.append(i - 1)
        else:
            # Check if any segments are too large
            prev = -1
            new_breakpoints = []
            for bp in breakpoints:
                if bp - prev > max_chunk_size:
                    # Add intermediate breakpoints
                    for i in range(prev + max_chunk_size, bp, max_chunk_size):
                        new_breakpoints.append(i)
                new_breakpoints.append(bp)
                prev = bp
                
            # Check the last segment
            if len(embeddings) - 1 - prev > max_chunk_size:
                for i in range(prev + max_chunk_size, len(embeddings) - 1, max_chunk_size):
                    new_breakpoints.append(i)
                    
            breakpoints = sorted(new_breakpoints)
        
        return breakpoints
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # This is a simplified approach - for production, consider using nltk or spacy
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_recursive_boundaries(self, text: str) -> List[str]:
        """
        Split text by natural boundaries like paragraphs and sections.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks split by natural boundaries
        """
        # First try splitting by multiple newlines (paragraphs/sections)
        if '\n\n\n' in text:
            return [chunk.strip() for chunk in text.split('\n\n\n') if chunk.strip()]
        
        # Then try double newlines (paragraphs)
        if '\n\n' in text:
            initial_splits = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            
            # Check if any splits are still too large
            result = []
            for split in initial_splits:
                if len(split) > self.chunk_size * 2:
                    # Try splitting large paragraphs by headings or bullet points
                    subsplits = self._split_by_headings_or_bullets(split)
                    result.extend(subsplits)
                else:
                    result.append(split)
            return result
        
        # If no paragraph breaks, try splitting by headings or bullet points
        heading_splits = self._split_by_headings_or_bullets(text)
        if len(heading_splits) > 1:
            return heading_splits
        
        # Last resort: return the whole text as one chunk
        return [text]
    
    def _split_by_headings_or_bullets(self, text: str) -> List[str]:
        """
        Split text by headings or bullet points.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks split by headings or bullets
        """
        # Try to identify heading patterns or bullet points
        heading_pattern = re.compile(r'\n[A-Z][^\n]{0,50}:\s*\n|\n\d+\.\s+[A-Z]|\n[•\-\*]\s+')
        
        splits = []
        last_end = 0
        
        for match in heading_pattern.finditer(text):
            # Don't split if match is at the beginning
            if match.start() > last_end:
                splits.append(text[last_end:match.start()])
                last_end = match.start()
        
        # Add the final chunk
        if last_end < len(text):
            splits.append(text[last_end:])
        
        # If we found meaningful splits, return them
        if len(splits) > 1:
            return [chunk.strip() for chunk in splits if chunk.strip()]
        
        # Otherwise return the original text
        return [text]
    
    def _sentence_splitting(self, text: str) -> List[str]:
        """
        Split text into sentences and combine into chunks under the target size.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Combine sentences into chunks under the target size
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Current chunk would exceed size limit, finalize it
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _basic_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method that splits text based on character count.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good end point (preferably at paragraph or sentence boundary)
            end = min(start + self.chunk_size, len(text))
            
            # Try to find paragraph break
            if end < len(text):
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + (self.chunk_size // 2):
                    end = paragraph_break + 2
            
            # If no paragraph break, try sentence break
            if end < len(text) and end == start + self.chunk_size:
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                if sentence_break != -1 and sentence_break > start + (self.chunk_size // 2):
                    end = sentence_break + 2
            
            # Get the chunk and add to list
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        logger.info(f"Basic chunking created {len(chunks)} chunks from text of length {len(text)}")
        return chunks