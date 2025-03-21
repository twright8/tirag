# Anti-Corruption RAG Configuration Guide

This guide explains how to use the central configuration system in the Anti-Corruption RAG project.

## Overview

The project uses a centralized configuration approach, where all configurable parameters are defined in a single source of truth: `config/config.py`. This design eliminates duplicate settings across files and ensures consistency throughout the application.

## Configuration Categories

The configuration file is organized into logical sections:

1. **System Paths and Directories**
   - Base paths for the project
   - Data storage locations
   - Model directories
   - Log paths

2. **Document Processing Settings**
   - OCR configuration
   - Supported file types
   - Processing parameters

3. **Chunking Settings**
   - Chunk size and overlap
   - Semantic chunking parameters

4. **Entity Extraction Settings**
   - Entity confidence thresholds
   - Entity type mappings
   - Relationship extraction parameters

5. **Coreference Resolution Settings**
   - Enable/disable toggle
   - Model selection
   - Batch processing size

6. **Model Settings**
   - NLP model paths
   - Embedding models
   - LLM configurations

7. **Hardware and Resource Settings**
   - GPU settings
   - Memory management thresholds
   - Model priorities and sizes

8. **Vector Database Settings**
   - Qdrant configuration
   - Distance metrics
   - Collection settings

9. **Search and Query Settings**
   - Hybrid search weights
   - Retrieval parameters
   - Query system settings

10. **UI Settings**
    - Theme configuration
    - Visualization parameters
    - Entity display formatting

11. **Exhaustive Extraction Settings**
    - Model extraction defaults
    - Batch sizes
    - Output formatting

## Using the Configuration in Code

To use the configuration in your code, follow these steps:

1. Import only the specific settings you need:

```python
from config.config import (
    ENTITY_CONFIDENCE_THRESHOLD,
    NER_MODEL,
    USE_GPU
)
```

2. Reference the settings in your code:

```python
def initialize_model():
    model = load_model(NER_MODEL)
    if USE_GPU:
        model.to('cuda')
    return model
```

## Modifying Configuration

When you need to modify configuration settings:

1. Edit only the central `config/config.py` file
2. Keep related settings grouped together
3. Add clear comments for any new settings
4. Use constants (ALL_CAPS) for all configuration parameters
5. Use type hints where possible to clarify expected values

## Configuration Through Streamlit UI

Many configuration parameters can be adjusted through the Streamlit UI at runtime. The UI accesses the same central configuration source, ensuring consistency between UI-based settings and code-based settings.

## Adding New Configuration Parameters

When adding new functionality that requires configuration:

1. Add the parameters to the appropriate section in `config/config.py`
2. Use descriptive names that make the purpose clear
3. Include default values that work in most situations
4. Add comments explaining the parameter's purpose
5. Consider adding the parameter to the UI if it's something users might want to adjust

## Best Practices

1. Never hardcode values that could be configurable
2. Always import from config.config rather than redefining values
3. Group logically related parameters in the configuration file
4. Use sensible defaults that work in most situations
5. Document any non-obvious configuration parameters
