"""
Coreference resolution module for improving document context.
"""
from typing import List, Dict, Any, Tuple
import re
import gc
import torch

from config.config import (
    ENABLE_COREFERENCE,
    COREFERENCE_MODEL,
    COREFERENCE_BATCH_SIZE
)
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage, free_memory

logger = setup_logger(__name__, "coreference_resolver.log")

class CoreferenceResolver:
    """
    Coreference resolution system using the Maverick model.
    """
    
    def __init__(self):
        """
        Initialize coreference resolver.
        """
        logger.info("Initializing CoreferenceResolver")
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        self.enabled = ENABLE_COREFERENCE
        self.batch_size = COREFERENCE_BATCH_SIZE
        
        logger.info(f"Coreference resolution {'enabled' if self.enabled else 'disabled'}")
        log_memory_usage(logger)
    
    def load_model(self):
        """
        Load the Maverick coreference model.
        """
        if self.model_loaded:
            return
        
        try:
            from maverick import Maverick
            
            logger.info("Loading Maverick coreference model")
            self.model = Maverick()
            self.model_loaded = True
            logger.info("Maverick model loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading Maverick model: {e}")
            raise
    
    def unload_model(self):
        """
        Unload the model to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info("Unloading Maverick coreference model")
            del self.model
            self.model = None
            self.model_loaded = False
            
            # Force garbage collection
            free_memory(logger)
            
            logger.info("Maverick model unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading Maverick model: {e}")
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks to resolve coreferences.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with resolved coreferences
        """
        logger.info(f"Processing {len(chunks)} chunks for coreference resolution")
        
        # If coreference resolution is disabled, return the original chunks
        if not self.enabled:
            logger.info("Coreference resolution is disabled. Returning original chunks.")
            return chunks
        
        # Ensure model is loaded
        if not self.model_loaded:
            self.load_model()
        
        # Process each chunk
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get the text
            text = chunk.get('text', '')
            
            # Skip empty chunks
            if not text.strip():
                processed_chunks.append(chunk)
                continue
            
            # Resolve coreferences
            resolved_text = self.resolve_coreferences(text)
            
            # Create new chunk with resolved text
            processed_chunk = chunk.copy()
            processed_chunk['text'] = resolved_text
            processed_chunk['metadata']['has_coref_resolution'] = True
            
            processed_chunks.append(processed_chunk)
            
            if (i + 1) % self.batch_size == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks for coreference resolution")
        
        logger.info(f"Completed coreference resolution for {len(chunks)} chunks")
        log_memory_usage(logger)
        
        return processed_chunks

    def resolve_coreferences(self, text: str) -> str:
        """
        Resolve coreferences in text using Maverick model.

        Args:
            text: Input text

        Returns:
            Text with resolved coreferences
        """
        try:
            # Process with Maverick model
            result = self.model.predict(text)

            # Debug the actual structure
            if isinstance(result, dict):

                # If result is a dictionary, check if it has the expected keys
                if 'clusters_token_offsets' in result and 'tokens' in result:
                    clusters = result['clusters_token_offsets']
                    tokens = result['tokens']
                    resolved_text = self.replace_coreferences_with_originals(tokens, clusters)
                    return resolved_text
                else:
                    logger.warning(f"Expected keys not found in Maverick result. Found: {list(result.keys())}")
                    return text
            elif isinstance(result, tuple) and len(result) == 2:
                # Original expected format
                clusters, tokens = result
                resolved_text = self.replace_coreferences_with_originals(tokens, clusters)
                return resolved_text
            else:
                logger.warning(f"Unexpected Maverick model return format: {type(result)}")
                return text  # Return original text for unexpected format

        except Exception as e:
            logger.error(f"Error during coreference resolution: {e}")
            logger.exception("Detailed traceback:")
            return text  # Return original text if there's an error
    def replace_coreferences_with_originals(self, tokens: List[str], clusters_token_offsets: List[List[Tuple[int, int]]]) -> str:
        """
        Replace coreferences with their antecedents, marking the replacements.
        
        Args:
            tokens: List of tokens from the text
            clusters_token_offsets: Clusters of coreference mentions
            
        Returns:
            Text with resolved coreferences
        """
        # Create a copy of tokens to modify
        modified_tokens = list(tokens)

        # Process clusters in reverse order to avoid index shifting problems
        for cluster in reversed(list(clusters_token_offsets)):
            if not cluster:
                continue

            # Get the first mention (antecedent) in the cluster
            antecedent_start, antecedent_end = cluster[0]
            antecedent = tokens[antecedent_start:antecedent_end + 1]
            antecedent_text = " ".join(antecedent)

            # Replace all subsequent mentions with the antecedent
            # Process in reverse order to maintain correct indices
            for mention_start, mention_end in reversed(cluster[1:]):
                original_mention = tokens[mention_start:mention_end + 1]
                original_text = " ".join(original_mention)

                # Check for special cases
                if len(original_mention) == 1:
                    mention = original_mention[0].lower()

                    # Handle contractions (he's, she's, they're, etc.)
                    if "'" in mention:
                        parts = mention.split("'")
                        if len(parts) == 2 and parts[1] in ["s", "re", "ve", "ll", "d"]:
                            # Handle different contractions
                            if parts[1] == "s":  # he's, she's (can be "is" or "has")
                                # For simplicity, we'll assume 's means "is"
                                replacement = [f"{antecedent_text} is", f"(#{original_text}#)"]
                            elif parts[1] == "re":  # they're
                                replacement = [f"{antecedent_text} are", f"(#{original_text}#)"]
                            elif parts[1] == "ve":  # they've
                                replacement = [f"{antecedent_text} have", f"(#{original_text}#)"]
                            elif parts[1] == "ll":  # they'll
                                replacement = [f"{antecedent_text} will", f"(#{original_text}#)"]
                            elif parts[1] == "d":  # they'd (can be "would" or "had")
                                replacement = [f"{antecedent_text} would", f"(#{original_text}#)"]
                            else:
                                replacement = antecedent + [f"(#{original_text}#)"]

                            modified_tokens[mention_start:mention_end + 1] = replacement
                            continue

                    # Handle possessive pronouns (his, her, their)
                    if mention in ["his", "her", "their", "hers", "theirs"]:
                        replacement = [f"{antecedent_text}'s", f"(#{original_text}#)"]
                        modified_tokens[mention_start:mention_end + 1] = replacement
                        continue

                # Default case: just replace with antecedent + original in brackets
                replacement = antecedent + [f"(#{original_text}#)"]
                modified_tokens[mention_start:mention_end + 1] = replacement

        # Join back into text
        text = " ".join(modified_tokens)

        # Clean up spacing around punctuation
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        return text