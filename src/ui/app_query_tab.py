"""
Query tab for the Anti-Corruption RAG system.
"""

import streamlit as st
from typing import Dict, Any

def render_query_tab(process_query, top_k=5):
    """Render the query tab with modern interface."""
    if not st.session_state.chunks:
        # Stylish empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📄</div>
            <h3 style="margin-bottom: 1rem; color: #1e293b;">No Documents Processed</h3>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto;">
                Upload and process documents using the sidebar to start analyzing your data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Two-column layout with chat history and input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Modern chat interface
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <h3 style="margin: 0 0 0.5rem 0; color: #1e293b;">Intelligence Assistant</h3>
                <p style="color: #64748b; margin: 0; font-size: 0.9rem;">
                    Ask questions about your documents to uncover hidden insights
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat container with fixed height
            chat_container = st.container()
            with chat_container:
                # Chat history
                if 'query_history' in st.session_state and st.session_state.query_history:
                    for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
                        query = item.get('query', '')
                        response = item.get('response', {})
                        answer = response.get('answer', 'No answer available.')
                        
                        # User message
                        st.markdown(f"""
                        <div style="display: flex; margin-bottom: 1rem;">
                            <div style="width: 40px; height: 40px; border-radius: 50%; background: #1e3a8a; 
                                     color: white; display: flex; align-items: center; justify-content: center;
                                     margin-right: 1rem; flex-shrink: 0;">
                                👤
                            </div>
                            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 8px; flex-grow: 1;">
                                {query}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Assistant message
                        st.markdown(f"""
                        <div style="display: flex; margin-bottom: 1rem;">
                            <div style="width: 40px; height: 40px; border-radius: 50%; background: #0ea5e9; 
                                     color: white; display: flex; align-items: center; justify-content: center;
                                     margin-right: 1rem; flex-shrink: 0;">
                                🤖
                            </div>
                            <div style="background: #f0f9ff; padding: 0.75rem; border-radius: 8px; flex-grow: 1;">
                                {answer}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Welcome message
                    st.markdown(f"""
                    <div style="display: flex; margin-bottom: 1rem;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background: #0ea5e9; 
                                 color: white; display: flex; align-items: center; justify-content: center;
                                 margin-right: 1rem; flex-shrink: 0;">
                            🤖
                        </div>
                        <div style="background: #f0f9ff; padding: 0.75rem; border-radius: 8px; flex-grow: 1;">
                            <p style="margin: 0 0 0.5rem 0;"><strong>Welcome to Anti-Corruption Intelligence!</strong></p>
                            <p style="margin: 0;">I can help you analyze your documents and uncover hidden relationships. Ask me anything about the content or entities found in your documents.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Query input with modern styling
            query = st.text_input(
                "Your question",
                key="query_input",
                placeholder="Ask about documents, entities, relationships..."
            )
            
            input_cols = st.columns([5, 1])
            with input_cols[0]:
                # Search parameters in a subtle dropdown
                with st.expander("Search options", expanded=False):
                    top_k = st.slider("Number of results", min_value=1, max_value=20, value=top_k)
                    use_reranking = st.checkbox("Use reranking", value=True)
            
            with input_cols[1]:
                search_button = st.button("Search", type="primary", use_container_width=True)
            
            if search_button and query:
                with st.spinner("Searching documents..."):
                    response = process_query(query, top_k=top_k)
                    # Force reload for UI update
                    st.rerun()
        
        with col2:
            # Right sidebar with source information
            if ('query_history' in st.session_state and 
                st.session_state.query_history and 
                'response' in st.session_state.query_history[-1]):
                
                last_response = st.session_state.query_history[-1]['response']
                
                # Source exploration panel
                st.markdown("""
                <div style="margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">Source Documents</h4>
                    <p style="color: #64748b; margin: 0; font-size: 0.8rem;">
                        Evidence supporting the answer
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Source documents
                for i, result in enumerate(last_response.get('context', [])[:5]):
                    with st.expander(f"Source {i+1}: {result.get('file_name', 'Unknown')}", expanded=i==0):
                        # Source metadata
                        st.markdown(f"""
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem;">
                            <span style="background: #f1f5f9; padding: 0.2rem 0.5rem; border-radius: 4px;">
                                Score: {result.get('score', 0):.3f}
                            </span>
                            {f" • Page {result.get('page_num', '')}" if result.get('page_num') else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Source text
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 0.75rem; border-radius: 6px; 
                                  font-size: 0.9rem; max-height: 200px; overflow-y: auto;">
                            {result.get('text', 'No text available.')}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Empty state for source panel
                st.markdown("""
                <div style="background: white; border-radius: 12px; padding: 1.5rem; 
                            text-align: center; height: 400px; display: flex; align-items: center; 
                            justify-content: center; flex-direction: column;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📄</div>
                    <p style="color: #64748b; margin: 0; font-size: 0.9rem;">
                        Source documents will appear here after your first query
                    </p>
                </div>
                """, unsafe_allow_html=True)