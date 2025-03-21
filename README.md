# Anti-Corruption Intelligence Suite

![Anti-Corruption Intelligence](https://img.shields.io/badge/Anti--Corruption-Intelligence-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xNSA0YTggOCAwIDEgMCAwIDE2IDggOCAwIDAgMCAwLTE2em0wIDJhNiA2IDAgMSAxIDAgMTIgNiA2IDAgMCAxIDAtMTJ6Ii8+PC9zdmc+)

A sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for anti-corruption document analysis. This cutting-edge tool combines advanced NLP techniques with an intuitive interface to help investigators uncover hidden relationships and insights in complex documents.

## Features

- **Advanced Document Processing Pipeline**: Support for PDF, Word, TXT, CSV/XLSX files with OCR capabilities
- **Semantic Chunking**: Intelligent document segmentation using multilingual embedding models
- **Coreference Resolution**: Enhanced context through pronoun resolution
- **Named Entity Recognition & Relationship Classification**: Advanced entity extraction and relationship mapping
- **BM25 & Embedding Indexing**: Hybrid search using both keyword and semantic matching
- **Conversational Query Interface**: Natural language interaction with VLLM integration
- **Resource Management**: Optimized CUDA and VRAM utilization with intelligent model loading/unloading
- **Modern Streamlit UI**: Beautiful, intuitive interface with visualizations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TI_RAG.git
   cd TI_RAG
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run setup to create directories and download required models:
   ```bash
   python setup.py
   ```

## Running the Application

Launch the Streamlit application with:
```bash
streamlit run app.py
```

The application will be accessible at http://localhost:8501 by default.

## System Usage

### Document Processing
1. Upload documents using the file uploader in the sidebar
2. Click "Process Documents" to initiate the pipeline
3. Monitor progress in the status indicators

### Querying Documents
1. Navigate to the "Query" tab
2. Enter your question in the text input
3. Adjust search parameters if needed
4. Click "Search" to get answers based on your documents

### Exploring Entities
1. Navigate to the "Entities" tab
2. Explore the interactive entity network visualization
3. Browse the entity list, filter by type, and sort as needed

### Managing Resources
- Monitor system resource usage in the sidebar
- Use resource management controls in case of memory issues
- Clear data or reset conversations as needed

## Technical Details

### Core Components

- **Document Processing**:
  - Document loading with OCR support
  - Semantic chunking
  - Coreference resolution

- **Entity Extraction**:
  - Named entity recognition using Flair
  - Relationship classification
  - Entity network graph construction

- **Indexing & Search**:
  - BM25 keyword indexing
  - Embedding-based semantic search
  - Reciprocal rank fusion for hybrid search

- **Query Processing**:
  - VLLM integration for fast inference
  - Context-aware response generation

### System Architecture

The system follows a modular design pattern with these key modules:

```
TI_RAG/
├── config/            # Configuration settings
├── src/
│   ├── document_processing/  # Document handling
│   ├── entity_extraction/    # NER and relationships
│   ├── indexing/             # BM25 and vector indices
│   ├── query_system/         # Search and LLM components
│   ├── ui/                   # Streamlit interface
│   └── utils/                # Common utilities
├── data/              # Data storage
│   ├── raw/           # Original documents
│   ├── processed/     # Processed documents
│   └── ocr_cache/     # OCR results cache
└── models/            # Model weights
```

## Memory Management

The system uses an intelligent resource manager to:
- Monitor RAM and VRAM usage
- Lazy-load models when needed
- Unload unused models to free memory
- Prioritize models based on usage patterns

## Extending the System

To add new capabilities:
1. Follow the modular design pattern
2. Keep files under 300 lines
3. Register new models with the ResourceManager
4. Update the UI to expose new features

## License

[MIT License](LICENSE)

## Acknowledgements

This project draws inspiration from the "raggle" reference project and combines multiple state-of-the-art techniques for document analysis.