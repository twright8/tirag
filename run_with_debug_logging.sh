#!/bin/bash
# Run the app with verbose logging to the console

# Configure logging to show in console
export PYTHONUNBUFFERED=1
export LOG_LEVEL=DEBUG
export LOG_TO_CONSOLE=true

# Set test text to examine closely
export TEST_TEXT_ENABLED=true

# Run the app
echo "Starting Anti-Corruption RAG with debug logging enabled..."
echo "This will show detailed logs in the console for troubleshooting."
echo ""

# Run streamlit
streamlit run app.py