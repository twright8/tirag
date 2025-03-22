#!/bin/bash
# Run the app with mock relationships enabled for testing

# Set the mock relationships environment variable
export MOCK_RELATIONSHIPS=true

# Run the app
echo "Starting Anti-Corruption RAG with mock relationships enabled..."
echo "This will generate random relationships between entities for testing the visualization."
echo ""

# Run streamlit
streamlit run app.py