"""
Entity extraction module for named entity recognition using Flair NER.
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import torch
import uuid
import Levenshtein
from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

from config.config import (
    ENTITY_TYPES, 
    USE_GPU, 
    GPU_ID, 
    NER_MODEL, 
    ENTITY_CONFIDENCE_THRESHOLD,
    ENTITY_TYPE_MAPPING, 
    UNWANTED_ENTITY_TYPES,
    ENTITY_DEDUPLICATION_THRESHOLD,
    ENTITY_BATCH_SIZE
)
from src.utils.logger import setup_logger, log_entity_extraction
from src.utils.resource_monitor import log_memory_usage, check_gpu_availability, free_memory

logger = setup_logger(__name__, "entity_extractor.log")

class EntityExtractor:
    """
    Entity extraction system using Flair NER.
    """
    
    def __init__(self, confidence_threshold: float = ENTITY_CONFIDENCE_THRESHOLD):
        """
        Initialize entity extractor.
        
        Args:
            confidence_threshold: Minimum confidence threshold for entities
        """
        self.entity_types = ENTITY_TYPES
        self.confidence_threshold = confidence_threshold
        self.ner_model_path = NER_MODEL
        self.batch_size = ENTITY_BATCH_SIZE
        
        # Entity type mapping for standardization
        self.entity_mapping = ENTITY_TYPE_MAPPING
        
        # Unwanted entity types to skip
        self.unwanted_types = UNWANTED_ENTITY_TYPES
        
        # Check GPU availability
        self.is_gpu_available, self.device = check_gpu_availability()
        self.device_id = GPU_ID if self.is_gpu_available and USE_GPU else -1
        
        # Initialize models
        self.ner_tagger = None
        self.sentence_splitter = None
        self.model_loaded = False
        
        logger.info(f"Initialized EntityExtractor with confidence_threshold={confidence_threshold}, "
                   f"GPU available: {self.is_gpu_available}")
        
        log_memory_usage(logger)
    
    def load_models(self):
        """
        Load the NER model and sentence splitter.
        """
        if self.model_loaded:
            return
        
        try:
            logger.info("Loading Flair NER model: flair/ner-english-ontonotes-fast")
            
            # Load sentence splitter
            self.sentence_splitter = SegtokSentenceSplitter()
            
            # Load NER tagger
            self.ner_tagger = Classifier.load(self.ner_model_path)
            
            # Move to GPU if available
            if self.is_gpu_available and USE_GPU:
                self.ner_tagger.to(torch.device(f'cuda:{self.device_id}'))
                logger.info(f"NER model loaded on GPU (device {self.device_id})")
            else:
                logger.info("NER model loaded on CPU")
            
            self.model_loaded = True
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            raise
    
    def unload_models(self):
        """
        Unload models to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info("Unloading NER models")
            
            # Delete models
            del self.ner_tagger
            del self.sentence_splitter
            self.ner_tagger = None
            self.sentence_splitter = None
            self.model_loaded = False
            
            # Force garbage collection
            free_memory(logger)
            
            logger.info("NER models unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading NER models: {e}")
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Process chunks to extract entities.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Tuple of (processed chunks, entity database)
        """
        logger.info(f"Processing {len(chunks)} chunks for entity extraction")
        
        # Ensure models are loaded
        if not self.model_loaded:
            self.load_models()
        
        # Process each chunk
        processed_chunks = []
        all_entities = {}
        
        for i, chunk in enumerate(chunks):
            # Get the text
            text = chunk.get('text', '')
            
            # Skip empty chunks
            if not text.strip():
                processed_chunks.append(chunk)
                continue
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Create new chunk with extracted entities
            processed_chunk = chunk.copy()
            processed_chunk['entities'] = entities
            
            # Mark entity mentions in the text
            marked_text = self.mark_entities_in_text(text, entities)
            processed_chunk['text_with_entities'] = marked_text
            
            # Add entities to the database
            doc_id = chunk.get('document_id', '')
            self._add_entities_to_database(entities, all_entities, doc_id, chunk.get('chunk_id', ''))
            
            processed_chunks.append(processed_chunk)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks for entity extraction")
        
        # Deduplicate and normalize entities
        entity_database = self._deduplicate_entities(all_entities)
        
        logger.info(f"Completed entity extraction for {len(chunks)} chunks, found {len(entity_database)} unique entities")
        log_memory_usage(logger)
        
        return processed_chunks, entity_database
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using Flair NER.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        try:
            # Split text into sentences
            flair_sentences = self.sentence_splitter.split(text)
            
            # Run NER on each sentence
            self.ner_tagger.predict(flair_sentences)
            
            # Process entities
            entities = []
            current_offset = 0
            
            for sentence in flair_sentences:
                # Get entities from the sentence
                for entity in sentence.get_spans('ner'):
                    # Get entity type and normalize it
                    entity_type = entity.tag.lower()
                    
                    # Skip unwanted entity types
                    if entity_type in self.unwanted_types:
                        continue
                    
                    # Map entity type to standard form
                    if entity_type in self.entity_mapping:
                        entity_type = self.entity_mapping[entity_type]
                    else:
                        # Skip entities with unknown types
                        continue
                    
                    # Skip entities with low confidence
                    if entity.score < self.confidence_threshold:
                        continue
                    
                    # Adjust positions to account for sentence offset in the original text
                    start_pos = entity.start_position + current_offset
                    end_pos = entity.end_position + current_offset
                    
                    # Create entity object
                    entity_obj = {
                        'id': str(uuid.uuid4()),
                        'text': entity.text,
                        'type': entity_type,
                        'confidence': entity.score,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    }
                    
                    entities.append(entity_obj)
                
                # Update offset for the next sentence
                current_offset += len(sentence.text) + 1  # +1 for the space/newline between sentences
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def mark_entities_in_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Mark entities in text with their types.
        
        Args:
            text: Original text
            entities: List of extracted entities
            
        Returns:
            Text with entity type markers
        """
        # Sort entities by start position (in reverse order to avoid index shifting)
        sorted_entities = sorted(entities, key=lambda e: e['start_pos'], reverse=True)
        
        # Mark entities in text
        marked_text = text
        for entity in sorted_entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            entity_type = entity['type']
            
            # Insert markers
            marked_text = (
                marked_text[:end_pos] + 
                f"##/{entity_type}##" +
                marked_text[end_pos:]
            )
            marked_text = (
                marked_text[:start_pos] + 
                f"##entity:{entity_type}##" +
                marked_text[start_pos:]
            )
        
        return marked_text
    
    def _add_entities_to_database(self, entities: List[Dict[str, Any]], 
                                entity_db: Dict[str, Dict[str, Any]],
                                document_id: str, chunk_id: str):
        """
        Add entities to the entity database.
        
        Args:
            entities: List of entities
            entity_db: Entity database to update
            document_id: Document ID
            chunk_id: Chunk ID
        """
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity['type']
            
            # Generate fingerprint (lowercase, normalized)
            entity_fingerprint = self._generate_entity_fingerprint(entity_text, entity_type)
            
            # Add to database if not exists or update if exists
            if entity_fingerprint not in entity_db:
                entity_db[entity_fingerprint] = {
                    'text': entity_text,
                    'type': entity_type,
                    'fingerprint': entity_fingerprint,
                    'mentions': [],
                    'similar_forms': set([entity_text.lower()]),
                    'highest_confidence': entity['confidence']
                }
            
            # Update similar forms and confidence
            entity_db[entity_fingerprint]['similar_forms'].add(entity_text.lower())
            entity_db[entity_fingerprint]['highest_confidence'] = max(
                entity_db[entity_fingerprint]['highest_confidence'],
                entity['confidence']
            )
            
            # Add mention
            mention = {
                'document_id': document_id,
                'chunk_id': chunk_id,
                'start_pos': entity['start_pos'],
                'end_pos': entity['end_pos'],
                'confidence': entity['confidence']
            }
            
            entity_db[entity_fingerprint]['mentions'].append(mention)
    
    def _generate_entity_fingerprint(self, entity_text: str, entity_type: str) -> str:
        """
        Generate a fingerprint for an entity.
        
        Args:
            entity_text: Entity text
            entity_type: Entity type
            
        Returns:
            Entity fingerprint
        """
        # Normalize text: lowercase, remove extra spaces
        normalized_text = " ".join(entity_text.lower().split())
        
        # Create fingerprint
        fingerprint = f"{normalized_text}|{entity_type}"
        
        return fingerprint
    
    def _deduplicate_entities(self, entity_db: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Deduplicate entities using fuzzy matching.
        
        Args:
            entity_db: Raw entity database
            
        Returns:
            Deduplicated entity database
        """
        logger.info(f"Deduplicating {len(entity_db)} entities")
        
        # Group entities by type for more efficient matching
        entities_by_type = {}
        for fingerprint, entity in entity_db.items():
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append((fingerprint, entity))
        
        # Deduplicate within each type
        merged_groups = {}
        canonical_forms = {}
        
        for entity_type, entities in entities_by_type.items():
            # Sort by number of mentions (higher first)
            sorted_entities = sorted(
                entities, 
                key=lambda x: (len(x[1]['mentions']), x[1]['highest_confidence']),
                reverse=True
            )
            
            # Initialize groups
            groups = []
            
            # Group similar entities
            for fingerprint, entity in sorted_entities:
                entity_text = entity['text'].lower()
                found_group = False
                
                # Check if this entity is similar to any existing group
                for group_idx, group in enumerate(groups):
                    # Check against canonical form
                    canonical_fp, canonical_entity = group[0]
                    canonical_text = canonical_entity['text'].lower()
                    
                    # Check similarity
                    if self._are_entities_similar(entity_text, canonical_text):
                        group.append((fingerprint, entity))
                        found_group = True
                        break
                
                # If no similar group found, create a new one
                if not found_group:
                    groups.append([(fingerprint, entity)])
            
            # For each group, find the canonical form and merge entities
            for group in groups:
                if len(group) == 1:
                    # Single entity, no need to merge
                    fingerprint, entity = group[0]
                    merged_groups[fingerprint] = entity
                    canonical_forms[fingerprint] = fingerprint
                else:
                    # Multiple entities, merge them
                    canonical_fp, canonical_entity = group[0]  # Use the most frequent entity as canonical
                    
                    # Create merged entity
                    merged_entity = {
                        'text': canonical_entity['text'],
                        'type': entity_type,
                        'fingerprint': canonical_fp,
                        'mentions': [],
                        'similar_forms': set(),
                        'highest_confidence': 0.0,
                        'variants': []
                    }
                    
                    # Merge data from all entities in the group
                    for fp, entity in group:
                        merged_entity['mentions'].extend(entity['mentions'])
                        merged_entity['similar_forms'].update(entity['similar_forms'])
                        merged_entity['highest_confidence'] = max(
                            merged_entity['highest_confidence'],
                            entity['highest_confidence']
                        )
                        
                        # Add variant if not the canonical form
                        if fp != canonical_fp:
                            merged_entity['variants'].append({
                                'text': entity['text'],
                                'fingerprint': fp,
                                'mentions_count': len(entity['mentions'])
                            })
                        
                        # Map this fingerprint to the canonical one
                        canonical_forms[fp] = canonical_fp
                    
                    # Convert set to list for serialization
                    merged_entity['similar_forms'] = list(merged_entity['similar_forms'])
                    
                    # Add merged entity to result
                    merged_groups[canonical_fp] = merged_entity
        
        logger.info(f"Deduplicated to {len(merged_groups)} entities")
        return merged_groups
    
    def _are_entities_similar(self, text1: str, text2: str, threshold: float = ENTITY_DEDUPLICATION_THRESHOLD) -> bool:
        """
        Check if two entity texts are similar using Levenshtein distance.
        
        Args:
            text1: First entity text
            text2: Second entity text
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if entities are similar, False otherwise
        """
        # Check for exact match first
        if text1 == text2:
            return True
        
        # Check if one is contained in the other
        if text1 in text2 or text2 in text1:
            return True
        
        # Calculate Levenshtein similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return True
        
        distance = Levenshtein.distance(text1, text2)
        similarity = 1.0 - (distance / max_len)
        
        return similarity >= threshold