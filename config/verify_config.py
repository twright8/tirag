"""
Configuration verification script.

This script validates the central configuration settings to ensure they are
properly defined and have reasonable values.
"""
import os
import sys
from pathlib import Path
import inspect

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the configuration
from config import config

def verify_config():
    """
    Verify that all required configuration parameters are present and valid.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    print("Verifying Anti-Corruption RAG configuration...")
    
    errors = []
    warnings = []
    
    # Check directory paths
    dirs_to_check = [
        ('BASE_DIR', config.BASE_DIR),
        ('DATA_DIR', config.DATA_DIR),
        ('MODEL_DIR', config.MODEL_DIR),
        ('LOG_DIR', config.LOG_DIR),
        ('RAW_DATA_DIR', config.RAW_DATA_DIR),
        ('PROCESSED_DATA_DIR', config.PROCESSED_DATA_DIR),
        ('OCR_CACHE_DIR', config.OCR_CACHE_DIR),
        ('EXPORT_DIR', config.EXPORT_DIR),
    ]
    
    for name, path in dirs_to_check:
        if not isinstance(path, Path):
            errors.append(f"{name} should be a Path object")
    
    # Check numeric parameters
    numeric_params = [
        ('CHUNK_SIZE', config.CHUNK_SIZE, 100, 2000),
        ('CHUNK_OVERLAP', config.CHUNK_OVERLAP, 0, 500),
        ('ENTITY_CONFIDENCE_THRESHOLD', config.ENTITY_CONFIDENCE_THRESHOLD, 0.0, 1.0),
        ('RELATIONSHIP_CONFIDENCE_THRESHOLD', config.RELATIONSHIP_CONFIDENCE_THRESHOLD, 0.0, 1.0),
        ('ENTITY_DEDUPLICATION_THRESHOLD', config.ENTITY_DEDUPLICATION_THRESHOLD, 0.0, 1.0),
        ('LLM_TEMPERATURE', config.LLM_TEMPERATURE, 0.0, 1.0),
        ('BM25_WEIGHT', config.BM25_WEIGHT, 0.0, 1.0),
        ('VECTOR_WEIGHT', config.VECTOR_WEIGHT, 0.0, 1.0),
        ('MAX_MEMORY_PERCENTAGE', config.MAX_MEMORY_PERCENTAGE, 10.0, 95.0),
        ('MAX_GPU_MEMORY_PERCENTAGE', config.MAX_GPU_MEMORY_PERCENTAGE, 10.0, 95.0),
    ]
    
    for name, value, min_val, max_val in numeric_params:
        if not isinstance(value, (int, float)):
            errors.append(f"{name} should be a number")
        elif value < min_val or value > max_val:
            warnings.append(f"{name} value ({value}) is outside recommended range ({min_val}-{max_val})")
    
    # Check boolean parameters
    boolean_params = [
        'OCR_ENABLED',
        'USE_RELATIONSHIP_EXTRACTION',
        'ENABLE_COREFERENCE',
        'USE_GPU',
        'DEEPSEEK_API_ENABLED',
        'DEEPSEEK_USE_REASONER'
    ]
    
    for name in boolean_params:
        if hasattr(config, name):
            value = getattr(config, name)
            if not isinstance(value, bool):
                errors.append(f"{name} should be a boolean")
        else:
            errors.append(f"Missing required boolean parameter: {name}")
    
    # Check string parameters
    string_params = [
        'EMBEDDING_MODEL_NAME',
        'NER_MODEL',
        'RERANKER_MODEL',
        'DEFAULT_LLM_MODEL',
        'QDRANT_HOST',
        'COREFERENCE_MODEL',
        'RELATIONSHIP_MODEL_PATH'
    ]
    
    for name in string_params:
        if hasattr(config, name):
            value = getattr(config, name)
            if not isinstance(value, str):
                errors.append(f"{name} should be a string")
            elif value.strip() == '':
                warnings.append(f"{name} is empty")
        else:
            errors.append(f"Missing required string parameter: {name}")
    
    # Check that weights sum to 1.0
    if hasattr(config, 'BM25_WEIGHT') and hasattr(config, 'VECTOR_WEIGHT'):
        weight_sum = config.BM25_WEIGHT + config.VECTOR_WEIGHT
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"BM25_WEIGHT + VECTOR_WEIGHT should sum to 1.0 (currently {weight_sum})")
    
    # Print results
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠️ Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("\n✅ Configuration valid! All parameters are properly defined.")
    elif not errors:
        print("\n✅ Configuration valid, but with warnings.")
    else:
        print("\n❌ Configuration invalid. Please fix the errors.")
    
    return len(errors) == 0

def list_all_config_params():
    """List all configuration parameters currently defined."""
    print("\nConfiguration parameters:")
    
    # Get all module attributes
    all_attrs = dir(config)
    
    # Filter to only upper case (constants)
    config_params = [attr for attr in all_attrs if attr.isupper()]
    
    # Group by category based on name prefix
    categories = {}
    
    for param in config_params:
        # Try to determine category from name
        category = None
        for prefix in ["CHUNK", "ENTITY", "OCR", "MODEL", "QDRANT", "BM25", "LLM",
                       "GPU", "MEMORY", "UI", "EMBEDDING", "DIRECTORY", "NER"]:
            if param.startswith(prefix):
                category = prefix
                break
        
        if category is None:
            category = "MISC"
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append(param)
    
    # Print by category
    for category, params in sorted(categories.items()):
        print(f"\n{category}:")
        for param in sorted(params):
            value = getattr(config, param)
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"  - {param} = {value_str}")

if __name__ == "__main__":
    is_valid = verify_config()
    
    if "--list" in sys.argv:
        list_all_config_params()
    
    sys.exit(0 if is_valid else 1)
