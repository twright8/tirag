"""
Streamlit application for Anti-Corruption RAG system.
"""
import streamlit as st
import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional
import threading
import networkx as nx
import pyvis.network as net
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import io
import base64

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import (
    STREAMLIT_THEME, RETRIEVE_TOP_K, RERANK_TOP_K
)
from src.document_processing.document_loader import DocumentLoader
from src.document_processing.document_chunker import DocumentChunker
from src.document_processing.coreference_resolver import CoreferenceResolver
from src.entity_extraction.entity_extractor import EntityExtractor
from src.entity_extraction.relationship_extractor import RelationshipExtractor
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.embedding_indexer import EmbeddingIndexer
from src.query_system.hybrid_searcher import HybridSearcher
from src.query_system.query_processor import QueryProcessor
from src.utils.resource_manager import ResourceManager
from src.utils.logger import setup_logger, get_cli_logger

logger = setup_logger(__name__, "streamlit_app.log")
cli_logger = get_cli_logger()

# Initialize session state dictionary with default values
def init_session_state():
    """Initialize all session state variables with default values."""
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False
    if 'documents' not in st.session_state:
        st.session_state['documents'] = []
    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = []
    if 'entities' not in st.session_state:
        st.session_state['entities'] = {}
    if 'relationships' not in st.session_state:
        st.session_state['relationships'] = []
    if 'processing_status' not in st.session_state:
        st.session_state['processing_status'] = None
    if 'processing_progress' not in st.session_state:
        st.session_state['processing_progress'] = 0
    if 'processing_message' not in st.session_state:
        st.session_state['processing_message'] = ""
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    if 'conversation_id' not in st.session_state:
        st.session_state['conversation_id'] = str(uuid.uuid4())
    if 'resource_status' not in st.session_state:
        st.session_state['resource_status'] = None
    if 'graph' not in st.session_state:
        st.session_state['graph'] = None

# Initialize components
@st.cache_resource
def init_components():
    """Initialize system components."""
    resource_manager = ResourceManager()
    resource_manager.start_monitoring()

    document_loader = DocumentLoader()
    document_chunker = DocumentChunker()
    coreference_resolver = CoreferenceResolver()
    entity_extractor = EntityExtractor()
    relationship_classifier = RelationshipExtractor()
    bm25_indexer = BM25Indexer()
    embedding_indexer = EmbeddingIndexer()
    hybrid_searcher = HybridSearcher(bm25_indexer, embedding_indexer)
    query_processor = QueryProcessor(hybrid_searcher)

    # Register models with resource manager
    if hasattr(document_chunker, 'load_embedding_model'):
        resource_manager.register_model(
            model_id="chunker_embedding",
            model_obj=None,
            model_size_gb=1.2,
            uses_gpu=True,
            priority=3,
            load_func=document_chunker.load_embedding_model,
            unload_func=document_chunker.unload_embedding_model
        )

    if hasattr(coreference_resolver, 'load_model'):
        resource_manager.register_model(
            model_id="coreference_model",
            model_obj=None,
            model_size_gb=1.5,
            uses_gpu=True,
            priority=2,
            load_func=coreference_resolver.load_model,
            unload_func=coreference_resolver.unload_model
        )

    if hasattr(entity_extractor, 'load_models'):
        resource_manager.register_model(
            model_id="ner_model",
            model_obj=None,
            model_size_gb=1.0,
            uses_gpu=True,
            priority=2,
            load_func=entity_extractor.load_models,
            unload_func=entity_extractor.unload_models
        )

    # Register the relationship extractor
    resource_manager.register_model(
        model_id="relation_model",
        model_obj=None,
        model_size_gb=1.0,
        uses_gpu=True,
        priority=2,
        load_func=relationship_classifier.load_models,
        unload_func=relationship_classifier.unload_models
    )

    if hasattr(embedding_indexer, 'load_model'):
        resource_manager.register_model(
            model_id="embedding_model",
            model_obj=None,
            model_size_gb=1.2,
            uses_gpu=True,
            priority=4,
            load_func=embedding_indexer.load_model,
            unload_func=embedding_indexer.unload_model
        )

    if hasattr(query_processor, 'load_model'):
        resource_manager.register_model(
            model_id="llm_model",
            model_obj=None,
            model_size_gb=6.0,
            uses_gpu=True,
            priority=5,
            load_func=query_processor.load_model,
            unload_func=query_processor.unload_model
        )

    return {
        'resource_manager': resource_manager,
        'document_loader': document_loader,
        'document_chunker': document_chunker,
        'coreference_resolver': coreference_resolver,
        'entity_extractor': entity_extractor,
        'relationship_classifier': relationship_classifier,
        'bm25_indexer': bm25_indexer,
        'embedding_indexer': embedding_indexer,
        'hybrid_searcher': hybrid_searcher,
        'query_processor': query_processor
    }

# Custom Streamlit styling
def apply_custom_styling():
    """Apply custom CSS styling for a better UI."""
    st.markdown("""
    <style>
    /* Base Styles */
    :root {
      --primary-color: #1e3a8a;
      --secondary-color: #0d9488;
      --accent-color: #d97706;
      --background-color: #f8f9fa;
      --card-bg-color: #ffffff;
      --text-color: #1e293b;
      --text-light: #94a3b8;
      --border-color: #e2e8f0;
      --shadow: 0 2px 4px rgba(0,0,0,0.05);
      --radius: 8px;
      --spacing-xs: 4px;
      --spacing-sm: 8px;
      --spacing-md: 16px;
      --spacing-lg: 24px;
      --spacing-xl: 32px;
      --spacing-xxl: 48px;
    }

    /* Typography */
    body {
      font-family: 'Inter', sans-serif;
      color: var(--text-color);
      background-color: var(--background-color);
    }

    h1, h2, h3, h4, h5, h6 {
      font-weight: 600;
      margin-bottom: var(--spacing-md);
    }

    code {
      font-family: 'Source Code Pro', monospace;
    }

    /* Layout */
    .main {
      padding: var(--spacing-xl);
    }

    .stApp {
      max-width: 1900px;
      margin: 0 auto;
    }

    /* Cards */
    div[data-testid="stExpander"] {
      background-color: var(--card-bg-color);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: var(--spacing-lg);
      margin-bottom: var(--spacing-lg);
    }

    /* Buttons */
    button[kind="primary"] {
      background-color: var(--primary-color);
      border-radius: 6px;
      font-weight: 600;
      text-transform: uppercase;
      padding: var(--spacing-sm) var(--spacing-md);
    }

    button[kind="secondary"] {
      background-color: transparent;
      border: 1px solid var(--secondary-color);
      color: var(--secondary-color);
      border-radius: 6px;
      font-weight: 600;
    }

    /* Inputs */
    input, select, textarea {
      border-radius: 6px;
      border: 1px solid var(--border-color);
      padding: var(--spacing-sm);
    }

    /* Progress Bars */
    div[role="progressbar"] > div {
      background-color: var(--primary-color);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
      background-color: var(--card-bg-color);
      padding: var(--spacing-md);
    }

    /* Custom components */
    .status-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: var(--spacing-xs);
    }

    .status-active {
      background-color: #10b981;
    }

    .status-inactive {
      background-color: #94a3b8;
    }

    .status-warning {
      background-color: #f59e0b;
    }

    .status-error {
      background-color: #dc2626;
    }

    /* Chat UI */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f4f9;
    }
    .chat-message.assistant {
        background-color: #f9f0f9;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    
    /* Entity card */
    .entity-card {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    .entity-type {
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
        color: #666;
    }
    .entity-confidence {
        font-size: 0.7rem;
        color: #777;
    }
    
    /* Entity colors */
    .person {
        border-left: 4px solid #3b82f6;
    }
    .organization {
        border-left: 4px solid #10b981;
    }
    .location {
        border-left: 4px solid #f59e0b;
    }
    .money {
        border-left: 4px solid #8b5cf6;
    }
    .law {
        border-left: 4px solid #ec4899;
    }
    .event {
        border-left: 4px solid #ef4444;
    }
    
    /* Improved tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        border-top: 2px solid #1e3a8a;
        border-right: 1px solid #e2e8f0;
        border-left: 1px solid #e2e8f0;
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Process documents with visible progress
def process_documents(uploaded_files):
    """Process uploaded documents through the pipeline."""
    if not uploaded_files:
        st.error("Please upload at least one document.")
        return

    components = init_components()
    document_loader = components['document_loader']
    document_chunker = components['document_chunker']
    coreference_resolver = components['coreference_resolver']
    entity_extractor = components['entity_extractor']
    relationship_classifier = components['relationship_classifier']
    bm25_indexer = components['bm25_indexer']
    embedding_indexer = components['embedding_indexer']

    # Update status and create progress displays
    st.session_state['processing_status'] = "running"
    st.session_state['processing_progress'] = 0
    st.session_state['processing_message'] = "Starting document processing..."
    
    # Create progress displays
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting document processing...")

    try:
        # Step 1: Document Loading
        documents = []
        status_text.text("Loading documents...")

        for i, uploaded_file in enumerate(uploaded_files):
            file_progress_text = st.empty()
            file_progress_text.text(f"Processing: {uploaded_file.name}")
            
            # Save to a temporary file
            temp_path = f"temp_{uuid.uuid4()}.{uploaded_file.name.split('.')[-1]}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load document
            document = document_loader.load_document(temp_path)
            documents.append(document)

            # Clean up temporary file
            os.remove(temp_path)
            
            file_progress_text.text(f"Completed: {uploaded_file.name}")

            # Update progress
            progress = (i + 1) / len(uploaded_files) * 20  # 20% for document loading
            st.session_state['processing_progress'] = progress
            progress_bar.progress(progress / 100)

        st.session_state['documents'] = documents
        st.session_state['processing_message'] = f"Loaded {len(documents)} documents. Chunking..."
        status_text.text(f"Loaded {len(documents)} documents. Chunking...")

        # Step 2: Document Chunking
        all_chunks = []
        for i, document in enumerate(documents):
            status_text.text(f"Chunking document {i+1}/{len(documents)}: {document.get('file_name', '')}")
            chunks = document_chunker.chunk_document(document)
            all_chunks.extend(chunks)

            # Update progress
            progress = 20 + (i + 1) / len(documents) * 20  # 20-40% for chunking
            st.session_state['processing_progress'] = progress
            progress_bar.progress(progress / 100)

        st.session_state['chunks'] = all_chunks
        status_text.text(f"Created {len(all_chunks)} chunks. Resolving coreferences...")

        # Step 3: Coreference Resolution
        status_text.text("Resolving coreferences...")
        processed_chunks = coreference_resolver.process_chunks(all_chunks)
        st.session_state['chunks'] = processed_chunks
        progress_bar.progress(0.4)  # 40% after coreference
        status_text.text("Coreference resolution complete. Extracting entities...")

        # Step 4: Entity Extraction
        status_text.text("Extracting entities...")
        processed_chunks, entity_db = entity_extractor.process_chunks(processed_chunks)
        st.session_state['chunks'] = processed_chunks
        st.session_state['entities'] = entity_db
        progress_bar.progress(0.6)  # 60% after entity extraction
        status_text.text(f"Extracted {len(entity_db)} entities. Classifying relationships...")

        # Step 5: Relationship Classification
        status_text.text("Classifying relationships...")
        relationships = relationship_classifier.extract_relationships(processed_chunks)
        st.session_state['relationships'] = relationships
        progress_bar.progress(0.7)  # 70% after relationship classification
        status_text.text(f"Classified {len(relationships)} relationships. Building graph...")

        # Step 6: Build Entity Graph
        status_text.text("Building entity graph...")
        graph = relationship_classifier.build_relationship_graph(entity_db, relationships)
        st.session_state['graph'] = graph
        progress_bar.progress(0.8)  # 80% after graph building
        status_text.text("Graph built. Indexing for search...")

        # Step 7: BM25 Indexing
        status_text.text("Building BM25 index...")
        bm25_indexer.add_chunks(processed_chunks)
        progress_bar.progress(0.9)  # 90% after BM25 indexing
        status_text.text("BM25 indexing complete. Adding to vector database...")

        # Step 8: Embedding Indexing
        status_text.text("Building vector embeddings...")
        embedding_indexer.add_chunks(processed_chunks)
        progress_bar.progress(1.0)  # 100% after embedding indexing
        status_text.text("✅ Processing complete! Ready to query.")

        # Update status
        st.session_state['processing_status'] = "complete"

    except Exception as e:
        st.session_state['processing_status'] = "error"
        st.session_state['processing_message'] = f"Error processing documents: {e}"
        status_text.text(f"❌ Error: {e}")
        logger.error(f"Error processing documents: {e}")
        logger.exception("Detailed traceback:")

# Query function
def process_query(query, top_k=5):
    """Process a user query."""
    if not query:
        return None

    # Add to history
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []

    # Check if there's a conversation ID
    if 'conversation_id' not in st.session_state:
        st.session_state['conversation_id'] = str(uuid.uuid4())

    components = init_components()
    query_processor = components['query_processor']

    # Process query
    response = query_processor.process_query(
        query=query,
        top_k=top_k,
        conversation_id=st.session_state['conversation_id']
    )

    # Add to history
    query_item = {
        'query': query,
        'response': response,
        'timestamp': time.time()
    }
    st.session_state['query_history'].append(query_item)

    return response

# Visualization functions
def visualize_entity_network():
    """Visualize entity network as an interactive graph."""
    if not st.session_state['graph']:
        st.warning("No entity relationships to visualize.")
        return

    # Create a wider pyvis network that fits better
    graph = st.session_state['graph']
    network = net.Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333", select_menu=True, filter_menu=True)

    # Set network options
    network.set_options("""
    {
        "nodes": {
            "shape": "dot",
            "size": 25,
            "font": {
                "size": 14,
                "face": "Arial"
            },
            "borderWidth": 2,
            "shadow": true
        },
        "edges": {
            "width": 2,
            "shadow": true,
            "smooth": {
                "type": "dynamic"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08
            },
            "stabilization": {
                "iterations": 100
            }
        }
    }
    """)

    # Define colors for entity types
    colors = {
        'person': '#3b82f6',       # Blue
        'organization': '#10b981', # Green
        'location': '#f59e0b',     # Amber
        'money': '#8b5cf6',        # Purple
        'law': '#ec4899',          # Pink
        'event': '#ef4444',        # Red
    }

    # Add nodes
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]

        # Get color based on entity type
        entity_type = node_data.get('type', 'unknown')
        color = colors.get(entity_type, '#94a3b8')  # Default gray

        # Add node
        network.add_node(
            node_id,
            label=node_data.get('label', 'Unknown'),
            title=f"Type: {entity_type}<br>Mentions: {node_data.get('mentions_count', 0)}",
            color=color,
            size=10 + (node_data.get('mentions_count', 1) * 2)  # Size based on mentions
        )

    # Add edges
    for source, target, edge_data in graph.edges(data=True):
        # Add edge
        network.add_edge(
            source,
            target,
            title=edge_data.get('type', 'associated_with'),
            label=edge_data.get('type', ''),
            value=edge_data.get('confidence', 0.5) * 5  # Width based on confidence
        )

    # Save and return HTML
    try:
        path = "entity_network.html"
        network.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()

        # Clean up
        if os.path.exists(path):
            os.remove(path)

        return html
    except Exception as e:
        logger.error(f"Error visualizing entity network: {e}")
        return "<p>Error generating visualization.</p>"

def create_entity_card(entity):
    """Create an HTML entity card for display."""
    entity_type = entity.get('type', 'unknown')
    mentions = entity.get('mentions', [])
    confidence = entity.get('highest_confidence', 0) * 100
    variants = entity.get('variants', [])

    html = f"""
    <div class="entity-card {entity_type}">
        <div class="entity-type">{entity_type}</div>
        <div class="entity-name">{entity.get('text', 'Unknown')}</div>
        <div class="entity-confidence">Confidence: {confidence:.1f}% | Mentions: {len(mentions)}</div>
    """

    if variants:
        html += '<div class="entity-variants">Also known as: '
        variant_texts = [v.get('text', 'Unknown') for v in variants[:3]]
        html += ", ".join(variant_texts)
        if len(variants) > 3:
            html += f" and {len(variants) - 3} more"
        html += '</div>'

    html += '</div>'
    return html

def display_resource_status():
    """Display resource usage status in a compact form."""
    components = init_components()
    resource_manager = components['resource_manager']

    # Get status
    status = resource_manager.get_resource_status()
    st.session_state['resource_status'] = status

    # Get key metrics
    ram = status.get('memory', {})
    gpu = status.get('gpu', {})
    models = status.get('models', {})
    
    # Create compact status display
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Memory bars
        ram_percent = ram.get('used_percent', 0)
        ram_used = ram.get('used_gb', 0)
        ram_total = ram.get('total_gb', 1)
        
        st.progress(ram_percent / 100, text=f"RAM: {ram_used:.1f}/{ram_total:.1f}GB")
        
        if gpu:
            gpu_percent = gpu.get('used_percent', 0)
            gpu_used = gpu.get('used_gb', 0)
            gpu_total = gpu.get('total_gb', 1)
            st.progress(gpu_percent / 100, text=f"GPU: {gpu_used:.1f}/{gpu_total:.1f}GB")
    
    with col2:
        # Model counts
        loaded_count = models.get('loaded_count', 0)
        total_count = models.get('total_count', 0)
        st.markdown(f"**Models:** {loaded_count}/{total_count}")
        
        # Quick actions
        if st.button("Unload All", use_container_width=True):
            resource_manager.unload_all_models()
            st.success("Models unloaded")

# Main UI
def main():
    """Main Streamlit application."""
    # Configure page for widescreen layout
    st.set_page_config(
        page_title="Anti-Corruption RAG",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Creative styling with modern aesthetics
    st.markdown("""
    <style>
    /* Force 16:10 aspect ratio with fullscreen experience */
    .main .block-container {
        max-width: none !important;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
        aspect-ratio: 16/10;
        overflow: auto;
    }
    
    /* Set sidebar width */
    [data-testid="stSidebarContent"] {
        min-width: 270px;
        max-width: 320px;
    }
    
    /* Modern color scheme */
    :root {
        --primary: #1e3a8a;
        --secondary: #0ea5e9;
        --accent: #f59e0b;
        --background: #f8fafc;
        --card: #ffffff;
        --text: #0f172a;
        --text-light: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    /* Beautiful gradients for progress bars */
    .stProgress > div > div {
        background-image: linear-gradient(to right, #0ea5e9, #1e3a8a) !important;
    }
    
    /* Custom header styling */
    h1, h2, h3 {
        color: var(--primary);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border-top: 3px solid #1e3a8a;
        color: #1e3a8a;
    }
    
    /* Card-like styling for expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    div.stButton > button:hover {
        background-color: #1e40af;
        border: none;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    /* Entity card styling */
    .entity-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 12px;
        margin-bottom: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .entity-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background-color: #f1f5f9;
        border-left: 4px solid #1e3a8a;
    }
    
    .chat-message.assistant {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
    }
    
    /* File uploader styling */
    .stFileUploader > div:first-child {
        padding-bottom: 0.5rem;
    }
    
    /* Text area styling */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Apply custom styling
    apply_custom_styling()

    # Initialize components if needed
    if not st.session_state['initialized']:
        components = init_components()
        st.session_state.initialized = True

    # Modern sidebar with visually appealing elements
    with st.sidebar:
        # Stylish logo/header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb;">
            <h1 style="margin: 0; font-size: 1.5rem; color: #1e3a8a;">
                <span style="font-size: 1.5rem; margin-right: 8px;">🔍</span> Intelligence Suite
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Document processing section with better visual hierarchy
        st.markdown("""
        <p style="font-weight: 600; font-size: 1rem; margin: 1rem 0 0.5rem 0; color: #1e293b;">
            Document Processing
        </p>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "xlsx"],
            label_visibility="collapsed"
        )

        # Visually distinct process button
        if uploaded_files:
            files_text = ", ".join([f.name for f in uploaded_files])
            st.markdown(f"""
            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.8rem; color: #475569;">
                    {len(uploaded_files)} file(s) ready
                </p>
                <p style="margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.75rem; color: #64748b;">
                    {files_text[:40]}{"..." if len(files_text) > 40 else ""}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files)
        else:
            st.markdown("""
            <div style="background: #f1f5f9; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
                <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                    <span style="font-size: 1.5rem; display: block; margin-bottom: 0.5rem;">📄</span>
                    Drag & drop files to begin
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Document stats if documents are processed
        if st.session_state.documents:
            st.markdown("""
            <p style="font-weight: 600; font-size: 1rem; margin: 1.5rem 0 0.5rem 0; border-top: 1px solid #e5e7eb; padding-top: 1rem; color: #1e293b;">
                Project Overview
            </p>
            """, unsafe_allow_html=True)
            
            # Stats in cards
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem;">
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;">
                    <p style="margin: 0; font-size: 1.25rem; font-weight: bold; color: #1e3a8a;">{len(st.session_state.documents)}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Documents</p>
                </div>
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;">
                    <p style="margin: 0; font-size: 1.25rem; font-weight: bold; color: #1e3a8a;">{len(st.session_state.chunks)}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Chunks</p>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem;">
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;">
                    <p style="margin: 0; font-size: 1.25rem; font-weight: bold; color: #1e3a8a;">{len(st.session_state.entities)}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Entities</p>
                </div>
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;">
                    <p style="margin: 0; font-size: 1.25rem; font-weight: bold; color: #1e3a8a;">{len(st.session_state.relationships)}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Relations</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions footer
        st.markdown("""
        <p style="font-weight: 600; font-size: 1rem; margin: 1.5rem 0 0.5rem 0; border-top: 1px solid #e5e7eb; padding-top: 1rem; color: #1e293b;">
            Quick Actions
        </p>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("🗑️ Clear Data", use_container_width=True):
                # Reset session state
                st.session_state.documents = []
                st.session_state.chunks = []
                st.session_state.entities = {}
                st.session_state.relationships = []
                st.session_state.graph = None
                
                # Clear indices
                components = init_components()
                bm25_indexer = components['bm25_indexer']
                embedding_indexer = components['embedding_indexer']
                
                bm25_indexer.clear_index()
                embedding_indexer.clear_collection()
                
                st.success("Data cleared")
                
        with cols[1]:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.conversation_id = str(uuid.uuid4())
                st.success("Chat reset")
        
        # Footer

    # Creative header with animation
    st.markdown("""
    <div style="background: linear-gradient(to right, #1e3a8a, #0ea5e9); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; display: flex; align-items: center;">
            <span style="font-size: 2.5rem; margin-right: 10px;">🔍</span>
            Anti-Corruption Intelligence
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Uncover hidden relationships in your documents with advanced NLP
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status in a clean card at the top
    if st.session_state.resource_status:
        ram = st.session_state.resource_status.get('memory', {})
        ram_percent = ram.get('used_percent', 0)
        
        gpu = st.session_state.resource_status.get('gpu', {})
        gpu_percent = gpu.get('used_percent', 0) if gpu else 0
        
        models = st.session_state.resource_status.get('models', {})
        loaded = models.get('loaded_count', 0)
        total = models.get('total_count', 0)
        
        status_cols = st.columns(4)
        with status_cols[0]:
            st.markdown(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <p style="color: #64748b; margin: 0; font-size: 0.8rem;">SYSTEM STATUS</p>
                <p style="margin: 0; font-weight: bold; font-size: 1.1rem; color: {'#10b981' if ram_percent < 80 else '#ef4444'};">
                    {'Healthy' if ram_percent < 80 else 'High Load'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_cols[1]:
            st.markdown(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <p style="color: #64748b; margin: 0; font-size: 0.8rem;">RAM USAGE</p>
                <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">{ram_percent:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with status_cols[2]:
            if gpu:
                st.markdown(f"""
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <p style="color: #64748b; margin: 0; font-size: 0.8rem;">GPU USAGE</p>
                    <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">{gpu_percent:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <p style="color: #64748b; margin: 0; font-size: 0.8rem;">GPU</p>
                    <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">Not Available</p>
                </div>
                """, unsafe_allow_html=True)
                
        with status_cols[3]:
            st.markdown(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <p style="color: #64748b; margin: 0; font-size: 0.8rem;">MODELS</p>
                <p style="margin: 0; font-weight: bold; font-size: 1.1rem;">{loaded}/{total} Active</p>
            </div>
            """, unsafe_allow_html=True)

    # Create modern spaced-out tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Query Documents", 
        "🕸️ Entity Network", 
        "📄 Document Explorer", 
        "⚙️ System Settings"
    ])

    # Tab 1: Query Interface - Using the modular query tab
    with tab1:
        from src.ui.app_query_tab import render_query_tab
        render_query_tab(process_query, RETRIEVE_TOP_K)

    # Tab 2: Entity Network
    with tab2:
        st.header("Entity Analysis")

        if not st.session_state.entities:
            st.info("No entities extracted yet. Please process documents first.")
        else:
            # Entity statistics
            entities = st.session_state.entities
            relationships = st.session_state.relationships

            st.subheader("Entity Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Entities", len(entities))

            with col2:
                st.metric("Total Relationships", len(relationships))

            with col3:
                # Count entity types
                entity_types = {}
                for entity_id, entity in entities.items():
                    entity_type = entity.get('type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

                # Find most common type
                most_common_type = max(entity_types.items(), key=lambda x: x[1])[0]
                st.metric("Most Common Entity Type", most_common_type)

            # Entity visualization
            st.subheader("Entity Network")

            if st.session_state.graph:
                html = visualize_entity_network()
                st.components.v1.html(html, height=600)

            # Entity list
            st.subheader("Entity List")

            # Filter entities
            entity_type_filter = st.multiselect(
                "Filter by entity type",
                options=list(set(entity.get('type', 'unknown') for entity in entities.values())),
                default=[]
            )

            # Sort entities
            sort_option = st.selectbox(
                "Sort by",
                options=["Mentions (High to Low)", "Confidence (High to Low)", "Alphabetically"]
            )

            # Display entities
            filtered_entities = []
            for entity_id, entity in entities.items():
                entity_type = entity.get('type', 'unknown')

                # Apply filter
                if entity_type_filter and entity_type not in entity_type_filter:
                    continue

                filtered_entities.append(entity)

            # Sort entities
            if sort_option == "Mentions (High to Low)":
                filtered_entities.sort(key=lambda e: len(e.get('mentions', [])), reverse=True)
            elif sort_option == "Confidence (High to Low)":
                filtered_entities.sort(key=lambda e: e.get('highest_confidence', 0), reverse=True)
            else:  # Alphabetical
                filtered_entities.sort(key=lambda e: e.get('text', '').lower())

            # Display entities in a grid
            cols = st.columns(3)
            for i, entity in enumerate(filtered_entities[:30]):  # Limit to 30 entities
                with cols[i % 3]:
                    st.markdown(create_entity_card(entity), unsafe_allow_html=True)

            if len(filtered_entities) > 30:
                st.info(f"Showing 30 of {len(filtered_entities)} entities. Use filters to narrow down.")

    # Tab 3: Document Overview
    with tab3:
        st.header("Document Overview")

        if not st.session_state.documents:
            st.info("No documents processed yet. Please upload and process documents first.")
        else:
            documents = st.session_state.documents
            chunks = st.session_state.chunks

            # Document statistics
            st.subheader("Document Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Documents", len(documents))

            with col2:
                st.metric("Total Chunks", len(chunks))

            with col3:
                # Calculate average chunks per document
                avg_chunks = len(chunks) / len(documents) if documents else 0
                st.metric("Avg. Chunks per Document", f"{avg_chunks:.1f}")

            # Document list
            st.subheader("Document List")

            for i, doc in enumerate(documents):
                doc_id = doc.get('document_id', f'Document {i+1}')
                file_name = doc.get('file_name', 'Unknown')
                file_type = doc.get('file_type', 'Unknown')

                # Extract metadata
                metadata = doc.get('metadata', {})
                page_count = metadata.get('page_count', 0)

                with st.expander(f"{file_name} ({file_type})", expanded=False):
                    # Display metadata
                    st.markdown("**Metadata:**")

                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.write(f"Document ID: {doc_id}")
                    with meta_cols[1]:
                        st.write(f"File Type: {file_type}")
                    with meta_cols[2]:
                        if 'page_count' in metadata:
                            st.write(f"Pages: {page_count}")
                        elif 'row_count' in metadata:
                            st.write(f"Rows: {metadata.get('row_count', 0)}")

                    # Show additional metadata
                    if 'title' in metadata and metadata['title']:
                        st.write(f"Title: {metadata['title']}")
                    if 'author' in metadata and metadata['author']:
                        st.write(f"Author: {metadata['author']}")

                    # Count document-specific chunks
                    doc_chunks = [c for c in chunks if c.get('document_id', '') == doc_id]
                    st.write(f"Chunks: {len(doc_chunks)}")

                    # Show sample chunks - fixed the nested expanders issue
                    if doc_chunks:
                        st.markdown("**Sample Chunks:**")
                        # Use columns instead of nested expanders
                        for j, chunk in enumerate(doc_chunks[:3]):  # Show only first 3 chunks
                            st.text(f"Chunk {j+1}:")
                            st.text_area(
                                label=f"Content",
                                value=chunk.get('text', 'No text available'),
                                height=100,
                                label_visibility="collapsed"
                            )

                        if len(doc_chunks) > 3:
                            st.info(f"Showing 3 of {len(doc_chunks)} chunks.")

    # Tab 4: Settings
    with tab4:
        st.header("System Settings")

        # System actions
        st.subheader("System Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear All Data"):
                # Reset session state
                st.session_state.documents = []
                st.session_state.chunks = []
                st.session_state.entities = {}
                st.session_state.relationships = []
                st.session_state.graph = None

                # Clear indices
                components = init_components()
                bm25_indexer = components['bm25_indexer']
                embedding_indexer = components['embedding_indexer']

                bm25_indexer.clear_index()
                embedding_indexer.clear_collection()

                st.success("All data has been cleared")

        with col2:
            if st.button("Reset Conversation"):
                st.session_state.query_history = []
                st.session_state.conversation_id = str(uuid.uuid4())
                st.success("Conversation has been reset")

        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            st.subheader("Model Settings")

            # LLM Model
            model_name = st.selectbox(
                "LLM Model",
                options=["Qwen/Qwen2.5-3B-Instruct", "Deepseek API"],
                index=0
            )

            if model_name == "Deepseek API":
                api_key = st.text_input("Deepseek API Key", type="password")
                use_reasoner = st.checkbox("Use Reasoner Mode", value=True)

            # Resource management
            st.subheader("Resource Management")

            col1, col2 = st.columns(2)

            with col1:
                max_memory = st.slider(
                    "Max Memory Usage (%)",
                    min_value=50,
                    max_value=95,
                    value=80
                )

            with col2:
                max_gpu = st.slider(
                    "Max GPU Usage (%)",
                    min_value=50,
                    max_value=95,
                    value=80
                )

            # Save settings
            if st.button("Save Settings"):
                # Here you would update the config
                st.success("Settings saved")

# Run the application
if __name__ == "__main__":
    main()