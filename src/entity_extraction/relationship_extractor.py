"""
Direct relationship extractor based on Flair.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import uuid
import torch
import networkx as nx
from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

from config.config import (
    RELATIONSHIP_CONFIDENCE_THRESHOLD,
    USE_RELATIONSHIP_EXTRACTION,
    RELATIONSHIP_MODEL_PATH,
    USE_GPU,
    GPU_ID,
    ENTITY_COLORS
)
from src.utils.logger import setup_logger
from src.utils.resource_monitor import log_memory_usage, check_gpu_availability, free_memory

logger = setup_logger(__name__, "relationship_extractor.log")

class RelationshipExtractor:
    """
    Relationship extraction using Flair's relation extraction model.
    This is a direct implementation based on the working example.
    """
    
    def __init__(self, confidence_threshold: float = RELATIONSHIP_CONFIDENCE_THRESHOLD):
        """Initialize relationship extractor"""
        self.confidence_threshold = confidence_threshold
        self.enabled = USE_RELATIONSHIP_EXTRACTION
        
        # Initialize models
        self.splitter = None
        self.tagger = None
        self.extractor = None
        self.is_loaded = False
        
        # Check GPU availability
        self.is_gpu_available, self.device = check_gpu_availability()
        logger.info(f"Initialized RelationshipExtractor with confidence_threshold={confidence_threshold}")
        log_memory_usage(logger)
    
    def load_models(self):
        """Load all required models"""
        if self.is_loaded:
            return
            
        try:
            logger.info("Loading relationship extraction models")
            
            # Create sentence splitter
            logger.info("Creating sentence splitter")
            self.splitter = SegtokSentenceSplitter()
            
            # Load NER tagger
            logger.info("Loading NER tagger 'flair/ner-english-ontonotes-fast'")
            self.tagger = Classifier.load("flair/ner-english-ontonotes-fast")
            if self.is_gpu_available and USE_GPU:
                self.tagger.to(torch.device(f'cuda:{GPU_ID}'))
                logger.info(f"NER tagger loaded on GPU (device {GPU_ID})")
            else:
                logger.info("NER tagger loaded on CPU")
                
            # Load relation extractor - DETAILED LOGGING FOR DEBUGGING
            logger.info("Loading relation extractor 'relations'")
            try:
                self.extractor = Classifier.load('relations')
                logger.info(f"Loaded relation extractor successfully. Type: {type(self.extractor)}")
            except Exception as e:
                logger.error(f"Failed to load 'relations' model: {e}")
                logger.exception("Detailed traceback for relations model loading:")
                
                # Try alternative model options
                try:
                    # Try from a directory if it exists
                    model_dir = os.path.join(os.environ.get('PROJECT_ROOT', '.'), 'models', 'relations')
                    if os.path.exists(model_dir):
                        logger.info(f"Trying to load relation model from directory: {model_dir}")
                        self.extractor = Classifier.load(model_dir)
                        logger.info("Loaded relation model from directory successfully")
                    else:
                        logger.error(f"Model directory not found: {model_dir}")
                        raise RuntimeError("Cannot find relation extraction model")
                except Exception as e2:
                    logger.error(f"Alternative loading also failed: {e2}")
                    raise
            
            if self.is_gpu_available and USE_GPU and self.extractor:
                try:
                    self.extractor.to(torch.device(f'cuda:{GPU_ID}'))
                    logger.info(f"Relation extractor loaded on GPU (device {GPU_ID})")
                except Exception as e:
                    logger.error(f"Error moving relation extractor to GPU: {e}")
                    logger.info("Keeping relation extractor on CPU")
                    
            self.is_loaded = True
            logger.info("All models loaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.exception("Detailed traceback:")
            raise
    
    def unload_models(self):
        """Unload all models to free memory"""
        if not self.is_loaded:
            return
            
        try:
            logger.info("Unloading relationship extraction models")
            
            # Delete models
            del self.splitter
            del self.tagger
            del self.extractor
            
            self.splitter = None
            self.tagger = None
            self.extractor = None
            self.is_loaded = False
            
            # Force garbage collection
            free_memory(logger)
            
            logger.info("Models unloaded successfully")
            log_memory_usage(logger)
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
    
    def extract_relationships(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships from document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of relationships
        """
        if not self.enabled:
            logger.info("Relationship extraction is disabled")
            return []
            
        logger.info(f"Extracting relationships from {len(chunks)} chunks")
        
        # Ensure models are loaded
        if not self.is_loaded:
            self.load_models()
        
        # Process each chunk
        all_relationships = []
        all_entities = {}
        
        # Use test text if enabled or if no chunks
        use_test_text = os.environ.get('TEST_TEXT_ENABLED', 'false').lower() == 'true'
        if not chunks or use_test_text:
            logger.warning("Using sample text for testing relationships")
            sample_text = """Jason is 30 years old and earns thirty thousand pounds per year.
He lives in London and works as a Software Engineer. His employee ID is 12345.
Susan, aged 9, does not have a job and earns zero pounds.
She is a student in elementary school and lives in Manchester.
Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
He resides in Birmingham and has an employee ID of 67890.
Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
She is based in Edinburgh and her employee ID is 54321."""
            test_chunk = {'document_id': 'test', 'chunk_id': 'test'}
            test_relationships, test_entities = self._process_chunk(sample_text, test_chunk)
            logger.info(f"Test processing resulted in {len(test_relationships)} relationships and {len(test_entities)} entities")
            
            # Only return test results if no chunks
            if not chunks:
                return test_relationships
        
        for i, chunk in enumerate(chunks):
            # Get text and process
            text = chunk.get('text', '')
            if not text:
                logger.warning(f"Chunk {i} has no text, skipping")
                continue
                
            # Log chunk info
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: text length {len(text)}")
            if len(text) > 100:
                logger.info(f"Chunk text preview: {text[:100]}...")
            else:
                logger.info(f"Chunk text: {text}")
                
            # Process the chunk
            chunk_relationships, chunk_entities = self._process_chunk(text, chunk)
            
            # Log results
            logger.info(f"Chunk {i+1} produced {len(chunk_relationships)} relationships and {len(chunk_entities)} entities")
            
            # Add relationships
            if chunk_relationships:
                all_relationships.extend(chunk_relationships)
                
            # Add entities to global map
            for entity_id, entity in chunk_entities.items():
                if entity_id not in all_entities:
                    all_entities[entity_id] = entity
            
            # Log progress
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                logger.info(f"Processed {i+1}/{len(chunks)} chunks for relationship extraction")
        
        # Deduplicate and clean up relationships
        unique_relationships = self._deduplicate_relationships(all_relationships)
        
        logger.info(f"Extracted {len(unique_relationships)} unique relationships out of {len(all_relationships)} total")
        return unique_relationships
    
    def _process_chunk(self, text: str, chunk_info: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Process a single chunk using the direct approach based on the working example
        
        Args:
            text: Text content of the chunk
            chunk_info: Metadata about the chunk
            
        Returns:
            Tuple of (relationships, entities)
        """
        try:
            # Split text into sentences
            sentences = self.splitter.split(text)
            logger.info(f"Split text into {len(sentences)} sentences")
            
            # Log first few sentences for debugging
            for i, sentence in enumerate(sentences[:3]):  # Show first 3 sentences
                if i < 3:  # Avoid excessive logging
                    logger.info(f"Sentence {i+1}: {sentence.text[:50]}...")
            
            # Run NER
            logger.info("Running NER tagging on sentences")
            self.tagger.predict(sentences)
            
            # Extract entities from NER results
            entities = {}
            entity_count = 0
            for sentence in sentences:
                sent_entities = sentence.get_spans('ner')
                entity_count += len(sent_entities)
                
                for entity in sent_entities:
                    # Create entity object
                    entity_id = str(uuid.uuid4())
                    entity_obj = {
                        'id': entity_id,
                        'text': entity.text,
                        'type': entity.tag.lower(),
                        'confidence': entity.score,
                        'document_id': chunk_info.get('document_id', ''),
                        'chunk_id': chunk_info.get('chunk_id', '')
                    }
                    entities[entity_id] = entity_obj
            
            logger.info(f"Found {entity_count} raw entities, {len(entities)} unique")
            
            # Log some entity examples
            entity_examples = list(entities.values())[:5]  # First 5 entities
            for i, entity in enumerate(entity_examples):
                logger.info(f"Entity {i+1}: {entity['text']} (type: {entity['type']}, confidence: {entity['confidence']:.4f})")
            
            # Run relation extraction
            logger.info("Running relation extraction on sentences")
            try:
                self.extractor.predict(sentences)
                logger.info("Relation extraction completed successfully")
            except Exception as e:
                logger.error(f"Error during relation extraction: {e}")
                logger.exception("Detailed traceback for relation extraction error:")
                return [], entities
            
            # Check and log relation results
            total_relations = 0
            for i, sentence in enumerate(sentences):
                rels = sentence.get_labels("relation")
                total_relations += len(rels)
                if rels and i < 5:  # Only log first 5 sentences with relations
                    logger.info(f"Sentence {i} has {len(rels)} relations")
                    for j, rel in enumerate(rels[:3]):  # First 3 relations per sentence
                        logger.info(f"  Relation {j+1}: {rel.value} (confidence: {rel.score:.4f})")
            
            logger.info(f"Found {total_relations} total relations across all sentences")
            
            # Extract relationships
            relationships = []
            
            # Build lookup maps
            entity_by_text = {}
            for entity_id, entity in entities.items():
                entity_text = entity['text'].lower()
                if entity_text not in entity_by_text:
                    entity_by_text[entity_text] = []
                entity_by_text[entity_text].append(entity)
            
            logger.info(f"Created entity lookup map with {len(entity_by_text)} distinct entity texts")
            
            # Process each sentence
            for sentence_idx, sentence in enumerate(sentences):
                # Get relation labels
                relation_labels = sentence.get_labels("relation")
                
                for rel_idx, rel_label in enumerate(relation_labels):
                    # Skip low confidence
                    if rel_label.score < self.confidence_threshold:
                        logger.debug(f"Skipping low confidence relation: {rel_label.score} < {self.confidence_threshold}")
                        continue
                    
                    # Parse relation text
                    relation_text = rel_label.value
                    logger.info(f"Processing relation: {relation_text}")
                    
                    try:
                        # NEW LOGIC - Handle direct relation type format
                        # The relation label is simply the relation type (e.g., "has_age")
                        if ":" not in relation_text and "/" not in relation_text:
                            logger.info(f"Found direct relation type: {relation_text}")
                            
                            # Extract the relationship based on entity types and positions
                            # We need to analyze the sentence to find the entities for this relation
                            rel_type = relation_text
                            
                            # Find entities in this sentence
                            sent_entities = []
                            for token_idx, token in enumerate(sentence.tokens):
                                # Check if this token is part of an entity
                                for entity_span in sentence.get_spans('ner'):
                                    if token_idx >= entity_span.start_position and token_idx <= entity_span.end_position:
                                        sent_entity = {
                                            'text': entity_span.text,
                                            'type': entity_span.tag.lower(),
                                            'start': entity_span.start_position,
                                            'end': entity_span.end_position
                                        }
                                        if sent_entity not in sent_entities:
                                            sent_entities.append(sent_entity)
                            
                            logger.info(f"Found {len(sent_entities)} entities in sentence: {[e['text'] for e in sent_entities]}")
                            
                            # If we have at least 2 entities, we can create a relationship
                            if len(sent_entities) >= 2:
                                # For simplicity, use the first two entities
                                source_entity_data = sent_entities[0]
                                target_entity_data = sent_entities[1]
                                
                                source_text = source_entity_data['text']
                                target_text = target_entity_data['text']
                                
                                logger.info(f"Using entities: {source_text} -> {target_text} for relation: {rel_type}")
                            else:
                                logger.warning(f"Not enough entities in sentence for relation: {rel_type}")
                                continue
                        else:
                            # ORIGINAL LOGIC - Try to parse the complex format
                            # Format: "Relation[0:1][2:5]: "Jason -> 30 years old"'/'has_age' (1.0)"
                            # Extract relation string
                            rel_parts = relation_text.split(": ")
                            if len(rel_parts) < 2:
                                logger.warning(f"Invalid relation format, no ':' delimiter: {relation_text}")
                                # Try to use it directly as the relation type
                                rel_type = relation_text
                                
                                # Try to find entities in the sentence
                                spans = sentence.get_spans('ner')
                                if len(spans) < 2:
                                    logger.warning(f"Not enough entities in sentence for relation: {rel_type}")
                                    continue
                                    
                                source_text = spans[0].text
                                target_text = spans[1].text
                                logger.info(f"Using entities from spans: {source_text} -> {target_text}")
                            else:
                                # Parse the complex format
                                entity_relation = rel_parts[1]
                                logger.debug(f"Entity relation part: {entity_relation}")
                                
                                main_parts = entity_relation.split("'/")
                                if len(main_parts) < 2:
                                    logger.warning(f"Invalid relation format, no '\\'/' delimiter: {entity_relation}")
                                    # Try alternative format
                                    main_parts = entity_relation.split("/")
                                    if len(main_parts) < 2:
                                        logger.warning(f"Alternative format also failed, no '/' delimiter")
                                        continue
                                
                                # Get entity text and relation type
                                entity_part = main_parts[0].strip('"').strip("'")
                                type_part = main_parts[1]
                                
                                logger.debug(f"Entity part: {entity_part}, Type part: {type_part}")
                                
                                # Extract relation type
                                if "'" in type_part:
                                    rel_type = type_part.split("'")[0]
                                else:
                                    rel_type = type_part.split(" ")[0]
                                
                                # Extract source and target entities
                                entity_parts = entity_part.split(" -> ")
                                if len(entity_parts) != 2:
                                    logger.warning(f"Invalid entity format, no ' -> ' delimiter: {entity_part}")
                                    continue
                                    
                                source_text, target_text = entity_parts
                                
                        logger.info(f"Extracted relation type: {rel_type}")
                        logger.info(f"Source: '{source_text}', Target: '{target_text}'")
                        logger.info(f"Source: '{source_text}', Target: '{target_text}'")
                        
                        # Find matching entities
                        source_entities = entity_by_text.get(source_text.lower(), [])
                        if not source_entities:
                            logger.warning(f"No exact match for source entity: {source_text}")
                            # Try partial match
                            for key, ents in entity_by_text.items():
                                if source_text.lower() in key or key in source_text.lower():
                                    source_entities = ents
                                    logger.info(f"Found partial match for source: {key}")
                                    break
                        
                        target_entities = entity_by_text.get(target_text.lower(), [])
                        if not target_entities:
                            logger.warning(f"No exact match for target entity: {target_text}")
                            # Try partial match
                            for key, ents in entity_by_text.items():
                                if target_text.lower() in key or key in target_text.lower():
                                    target_entities = ents
                                    logger.info(f"Found partial match for target: {key}")
                                    break
                        
                        if not source_entities or not target_entities:
                            logger.warning(f"Could not find entities: source={bool(source_entities)}, target={bool(target_entities)}")
                            
                            # Create artificial entities for missing ones
                            if not source_entities:
                                # Create source entity
                                source_id = str(uuid.uuid4())
                                source_entity = {
                                    'id': source_id,
                                    'text': source_text,
                                    'type': 'person',  # Default type
                                    'confidence': 0.9,
                                    'document_id': chunk_info.get('document_id', ''),
                                    'chunk_id': chunk_info.get('chunk_id', '')
                                }
                                # Add to entities
                                entities[source_id] = source_entity
                                entity_by_text[source_text.lower()] = [source_entity]
                                logger.info(f"Created artificial source entity: {source_text}")
                            else:
                                source_entity = source_entities[0]
                                
                            if not target_entities:
                                # Determine type
                                target_type = 'unknown'
                                if rel_type == 'has_age':
                                    target_type = 'number'
                                elif rel_type == 'lived_in' or rel_type == 'based_in':
                                    target_type = 'location'
                                elif rel_type == 'has_title':
                                    target_type = 'title'
                                    
                                # Create target entity
                                target_id = str(uuid.uuid4())
                                target_entity = {
                                    'id': target_id,
                                    'text': target_text,
                                    'type': target_type,
                                    'confidence': 0.9,
                                    'document_id': chunk_info.get('document_id', ''),
                                    'chunk_id': chunk_info.get('chunk_id', '')
                                }
                                # Add to entities
                                entities[target_id] = target_entity
                                entity_by_text[target_text.lower()] = [target_entity]
                                logger.info(f"Created artificial target entity: {target_text}")
                            else:
                                target_entity = target_entities[0]
                        else:
                            source_entity = source_entities[0]
                            target_entity = target_entities[0]
                        
                        # Create relationship
                        relationship = {
                            'id': str(uuid.uuid4()),
                            'source_id': source_entity['id'],
                            'source': source_entity['text'],
                            'source_type': source_entity['type'],
                            'target_id': target_entity['id'],
                            'target': target_entity['text'],
                            'target_type': target_entity['type'],
                            'relation_type': rel_type,
                            'confidence': rel_label.score,
                            'document_id': chunk_info.get('document_id', ''),
                            'chunk_id': chunk_info.get('chunk_id', '')
                        }
                        
                        logger.info(f"Created relationship: {source_entity['text']} --({rel_type})--> {target_entity['text']}")
                        relationships.append(relationship)
                        
                    except Exception as e:
                        logger.error(f"Error parsing relation: {e}")
                        logger.exception(f"Detailed traceback for relation {relation_text}:")
                        continue
            
            logger.info(f"Created {len(relationships)} relationships from chunk")
            return relationships, entities
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            logger.exception("Detailed traceback:")
            return [], {}
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships
        
        Args:
            relationships: List of relationships
            
        Returns:
            Deduplicated list
        """
        if not relationships:
            return []
            
        # Group by source, target, and relation type
        rel_groups = {}
        
        for rel in relationships:
            key = f"{rel['source_id']}|{rel['relation_type']}|{rel['target_id']}"
            
            if key not in rel_groups:
                rel_groups[key] = []
                
            rel_groups[key].append(rel)
        
        # Take highest confidence relationship from each group
        unique_rels = []
        
        for group in rel_groups.values():
            # Sort by confidence and take the highest
            sorted_group = sorted(group, key=lambda r: r['confidence'], reverse=True)
            unique_rels.append(sorted_group[0])
        
        return unique_rels
        
    def build_relationship_graph(self, entities: Dict[str, Dict[str, Any]], 
                               relationships: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a directed graph of entities and relationships
        
        Args:
            entities: Dictionary of entities by ID
            relationships: List of relationships
            
        Returns:
            NetworkX DiGraph
        """
        logger.info(f"Building relationship graph with {len(entities)} entities and {len(relationships)} relationships")
        
        # Create graph
        G = nx.DiGraph()
        
        # Add entities as nodes
        for entity_id, entity in entities.items():
            entity_type = entity.get('type', 'unknown')
            color = ENTITY_COLORS.get(entity_type, ENTITY_COLORS.get('default', '#94a3b8'))
            
            G.add_node(
                entity_id,
                label=entity['text'],
                type=entity_type,
                color=color,
                confidence=entity.get('confidence', 0.0)
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
                confidence=rel['confidence']
            )
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G