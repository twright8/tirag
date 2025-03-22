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
            
        # No debug modes - use the real relation extractor
        
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