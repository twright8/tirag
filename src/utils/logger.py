"""
Logging setup for the Anti-Corruption RAG system.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Set up logging directory
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
os.makedirs(log_dir, exist_ok=True)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_path = log_dir / log_file if not os.path.isabs(log_file) else log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_entity_extraction(logger: logging.Logger, text: str, entities: list) -> None:
    """
    Log entity extraction results in a readable format.
    
    Args:
        logger: Logger to use
        text: Original text
        entities: Extracted entities
    """
    if not entities:
        logger.info("No entities found in text")
        return
    
    entity_summary = "\n".join([
        f"- {entity['type'].upper()}: {entity['text']} (confidence: {entity['confidence']:.2f})"
        for entity in entities
    ])
    
    logger.info(f"Found {len(entities)} entities:\n{entity_summary}")

def log_relationship_extraction(logger: logging.Logger, relationships: list) -> None:
    """
    Log relationship extraction results in a readable format.
    
    Args:
        logger: Logger to use
        relationships: Extracted relationships
    """
    if not relationships:
        logger.info("No relationships found")
        return
    
    relationship_summary = "\n".join([
        f"- {rel['source']} ({rel['source_type']}) -> "
        f"{rel['relation_type']} -> "
        f"{rel['target']} ({rel['target_type']}) "
        f"(confidence: {rel['confidence']:.2f})"
        for rel in relationships
    ])
    
    logger.info(f"Found {len(relationships)} relationships:\n{relationship_summary}")
    
def log_query_result(logger: logging.Logger, query: str, results: list, execution_time: float) -> None:
    """
    Log query results in a readable format.
    
    Args:
        logger: Logger to use
        query: Query string
        results: Query results
        execution_time: Query execution time in seconds
    """
    logger.info(f"Query: \"{query}\"")
    logger.info(f"Found {len(results)} results in {execution_time:.2f} seconds")
    
    if results:
        top_result = results[0]
        logger.info(f"Top result: {top_result.get('document_id', 'unknown')} - "
                   f"Score: {top_result.get('score', 0):.4f}")
    
def get_cli_logger() -> logging.Logger:
    """
    Get a logger specifically for CLI output with minimal formatting.
    
    Returns:
        logging.Logger: CLI logger
    """
    logger = logging.getLogger("cli")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Simple formatter without timestamps for cleaner CLI output
    formatter = logging.Formatter('%(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger