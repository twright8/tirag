#!/bin/bash
# Run the app with sample relationship data for testing

# Set the sample relationship environment variable
export SAMPLE_REL_DATA=true

# Run the app
echo "Starting Anti-Corruption RAG with sample relationship data..."
echo "This will use the example relationship data to create a visualization."
echo ""

# Run streamlit
streamlit run app.py