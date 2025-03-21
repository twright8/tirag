"""
Relationship classification module using Flair's relation extraction.
"""
from typing import List, Dict, Any, Tuple, Set, Optional
import uuid
import networkx as nx
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from flair.data import Sentence

from config.config import (
    RELATIONSHIP_CONFIDENCE_THRESHOLD,
    USE_RELATIONSHIP_EXTRACTION,
    RELATIONSHIP_MODEL_PATH,
    USE_GPU,
    GPU_ID,
    ENTITY_COLORS
)
from src.utils.logger import setup_logger, log_relationship_extraction
from src.utils.resource_monitor import log_memory_usage, check_gpu_availability, free_memory

logger = setup_logger(__name__, "relationship_classifier.log")

class RelationshipClassifier:
    """
    Relationship classifier using Flair's relation extraction model.
    """
    
    def __init__(self, confidence_threshold: float = RELATIONSHIP_CONFIDENCE_THRESHOLD):
        """
        Initialize relationship classifier.
        
        Args:
            confidence_threshold: Minimum confidence threshold for relationships
        """
        self.confidence_threshold = confidence_threshold
        self.enabled = USE_RELATIONSHIP_EXTRACTION
        self.model_path = RELATIONSHIP_MODEL_PATH
        
        # Initialize models
        self.relation_extractor = None
        self.sentence_splitter = None
        self.model_loaded = False
        
        # Check GPU availability
        self.is_gpu_available, self.device = check_gpu_availability()
        
        logger.info(f"Initialized RelationshipClassifier with confidence_threshold={confidence_threshold}, enabled={self.enabled}")
        log_memory_usage(logger)
    
    def load_models(self):
        """
        Load the relationship extraction model.
        """
        if self.model_loaded:
            return
        
        try:
            logger.info("Loading Flair relation extraction model")
            
            # Load sentence splitter
            self.sentence_splitter = SegtokSentenceSplitter()
            
            # Load relation extractor
            self.relation_extractor = Classifier.load(self.model_path)
            
            # Move to GPU if available
            if self.is_gpu_available and USE_GPU:
                self.relation_extractor.to(self.device)
                logger.info(f"Relation extraction model loaded on GPU")
            else:
                logger.info("Relation extraction model loaded on CPU")
            
            self.model_loaded = True
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading relation extraction model: {e}")
            raise
    
    def unload_models(self):
        """
        Unload models to free memory.
        """
        if not self.model_loaded:
            return
        
        try:
            logger.info("Unloading relation extraction models")
            
            # Delete models
            del self.relation_extractor
            del self.sentence_splitter
            self.relation_extractor = None
            self.sentence_splitter = None
            self.model_loaded = False
            
            # Force garbage collection
            free_memory(logger)
            
            logger.info("Relation extraction models unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading relation extraction models: {e}")
    
    def extract_relationships(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships from chunks using Flair's relation extraction.
        
        Args:
            chunks: List of document chunks with extracted entities
            
        Returns:
            List of extracted relationships
        """
        logger.info(f"Extracting relationships from {len(chunks)} chunks")
        
        # Skip if relationship extraction is disabled
        if not self.enabled:
            logger.info("Relationship extraction is disabled. Skipping.")
            return []
        
        # Ensure models are loaded
        if not self.model_loaded:
            self.load_models()
        
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            # Skip chunks without entities
            if 'entities' not in chunk or not chunk['entities']:
                continue
            
            # Get the text
            text = chunk.get('text', '')
            
            # Split into sentences
            sentences = self.sentence_splitter.split(text)
            
            # Extract relationships from each sentence
            chunk_relationships = self._process_sentences(sentences, chunk)
            
            if chunk_relationships:
                all_relationships.extend(chunk_relationships)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks for relationship extraction")
        
        # Remove duplicates
        unique_relationships = self._deduplicate_relationships(all_relationships)
        
        logger.info(f"Extracted {len(unique_relationships)} unique relationships from {len(chunks)} chunks")
        log_memory_usage(logger)
        
        return unique_relationships
    
    def _process_sentences(self, sentences: List[Sentence], chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process sentences to extract relationships.
        
        Args:
            sentences: List of Flair sentences
            chunk: Chunk data for context
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Extract entities from the chunk
        entities = chunk.get('entities', [])
        
        # Create a mapping of entity positions for lookup
        entity_by_position = {}
        for entity in entities:
            start_pos = entity['start_pos']
            end_pos = entity['end_pos']
            entity_by_position[(start_pos, end_pos)] = entity
        
        # Predict relationships
        self.relation_extractor.predict(sentences)
        
        # Extract relationships from predictions
        for sent_idx, sentence in enumerate(sentences):
            relation_labels = sentence.get_labels("relation")
            
            for rel_label in relation_labels:
                # Skip relationships with low confidence
                if rel_label.score < self.confidence_threshold:
                    continue
                
                # Extract relationship components
                relation_text = rel_label.value
                relation_type = rel_label.data_point.value
                relation_confidence = rel_label.score
                
                # Parse relationship indices from the label text
                # Format: "Relation[0:1][2:5]: "Jason -> 30 years old"'/'has_age' (1.0)"
                try:
                    # Extract source and target positions from the relation span
                    span_info = relation_text.split(":")[0]  # "Relation[0:1][2:5]"
                    source_span = span_info.split("][")[0].replace("Relation[", "")  # "0:1"
                    target_span = span_info.split("][")[1].replace("]", "")  # "2:5"
                    
                    source_start, source_end = map(int, source_span.split(":"))
                    target_start, target_end = map(int, target_span.split(":"))
                    
                    # Get the corresponding text
                    source_text = sentence.tokens[source_start:source_end+1].text
                    target_text = sentence.tokens[target_start:target_end+1].text
                    
                    # Calculate offset in the original text
                    sentence_start_pos = text.find(sentence.text)
                    if sentence_start_pos == -1:
                        continue
                    
                    # Find matching entities based on text matching
                    source_entity = None
                    target_entity = None
                    
                    # Iterate through entities to find matches
                    for entity in entities:
                        entity_text = entity['text']
                        
                        if entity_text == source_text and source_entity is None:
                            source_entity = entity
                        elif entity_text == target_text and target_entity is None:
                            target_entity = entity
                    
                    # Skip if we couldn't find both entities
                    if not source_entity or not target_entity:
                        continue
                    
                    # Create relationship
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'source_id': source_entity['id'],
                        'source': source_entity['text'],
                        'source_type': source_entity['type'],
                        'target_id': target_entity['id'],
                        'target': target_entity['text'],
                        'target_type': target_entity['type'],
                        'relation_type': relation_type,
                        'confidence': relation_confidence,
                        'document_id': chunk.get('document_id', ''),
                        'chunk_id': chunk.get('chunk_id', ''),
                        'context': self._extract_context(sentence.text)
                    }
                    
                    relationships.append(relationship)
                    
                except Exception as e:
                    logger.error(f"Error parsing relationship: {e} - {relation_text}")
                    continue
        
        return relationships
    
    def _extract_context(self, text: str, window_size: int = 100) -> str:
        """
        Extract a context snippet around a relationship.
        
        Args:
            text: Full text of the sentence
            window_size: Maximum length of the context snippet
            
        Returns:
            Context text
        """
        if len(text) <= window_size:
            return text
        
        # Truncate and add ellipsis if needed
        return text[:window_size] + "..."
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships.
        
        Args:
            relationships: List of extracted relationships
            
        Returns:
            List of unique relationships
        """
        if not relationships:
            return []
        
        # Group by source, target, and relation type
        relationship_groups = {}
        
        for rel in relationships:
            key = f"{rel['source_id']}|{rel['relation_type']}|{rel['target_id']}"
            
            if key not in relationship_groups:
                relationship_groups[key] = []
            
            relationship_groups[key].append(rel)
        
        # Keep the highest confidence relationship from each group
        unique_relationships = []
        
        for group in relationship_groups.values():
            best_rel = max(group, key=lambda r: r['confidence'])
            unique_relationships.append(best_rel)
        
        return unique_relationships
    
    def build_entity_graph(self, entities: Dict[str, Dict[str, Any]], 
                          relationships: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a directed graph of entities and relationships.
        
        Args:
            entities: Entity database
            relationships: List of relationships
            
        Returns:
            NetworkX DiGraph
        """
        logger.info(f"Building entity graph with {len(entities)} entities and {len(relationships)} relationships")
        
        # Create graph
        G = nx.DiGraph()
        
        # Add entities as nodes
        for entity_id, entity in entities.items():
            entity_type = entity['type']
            color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS['default'])
            
            G.add_node(
                entity_id,
                label=entity['text'],
                type=entity_type,
                color=color,
                mentions_count=len(entity.get('mentions', [])),
                confidence=entity.get('highest_confidence', 0.0)
            )
        
        # Add relationships as edges
        for rel in relationships:
            source_id = rel['source_id']
            target_id = rel['target_id']
            
            # Skip if either entity is not in the graph
            if source_id not in G.nodes or target_id not in G.nodes:
                continue
            
            G.add_edge(
                source_id,
                target_id,
                type=rel['relation_type'],
                confidence=rel['confidence'],
                context=rel.get('context', '')
            )
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G