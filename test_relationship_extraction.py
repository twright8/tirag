"""
Test script for relationship extraction.
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

# Set up environment for testing
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['LOG_TO_CONSOLE'] = 'true'

# Import required components
from src.entity_extraction.relationship_extractor import RelationshipExtractor
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.DEBUG)

def main():
    """Test relationship extraction on sample text."""
    # Sample text from your working example
    sample_text = """Jason is 30 years old and earns thirty thousand pounds per year.
He lives in London and works as a Software Engineer. His employee ID is 12345.
Susan, aged 9, does not have a job and earns zero pounds.
She is a student in elementary school and lives in Manchester.
Michael, a 45-year-old Doctor, earns ninety-five thousand pounds annually.
He resides in Birmingham and has an employee ID of 67890.
Emily is a 28-year-old Data Scientist who earns seventy-two thousand pounds.
She is based in Edinburgh and her employee ID is 54321."""

    # Create a chunk
    chunk = {
        'document_id': 'test_doc',
        'chunk_id': 'test_chunk',
        'text': sample_text
    }
    
    # Create the relationship extractor
    extractor = RelationshipExtractor()
    
    # Extract relationships
    logger.info("Testing relationship extraction...")
    relationships = extractor.extract_relationships([chunk])
    
    # Print results
    logger.info(f"Found {len(relationships)} relationships")
    
    for i, rel in enumerate(relationships):
        logger.info(f"Relationship {i+1}:")
        logger.info(f"  Source: {rel['source']} ({rel['source_type']})")
        logger.info(f"  Relation: {rel['relation_type']}")
        logger.info(f"  Target: {rel['target']} ({rel['target_type']})")
        logger.info(f"  Confidence: {rel['confidence']}")
    
    # Build a graph
    entities = {}
    graph = extractor.build_relationship_graph(entities, relationships)
    logger.info(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

if __name__ == "__main__":
    main()
