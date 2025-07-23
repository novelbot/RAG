"""
Simple Query Interface - Streamlined and functional version
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional
from webui.auth import require_auth


@require_auth
def show():
    """Display a simplified, working query interface"""
    st.title("🔍 RAG Query Interface")
    
    # Initialize session state for chat history
    if "simple_chat_messages" not in st.session_state:
        st.session_state.simple_chat_messages = []
    
    # Get auth headers from session state
    from webui.auth import AuthManager
    auth_manager = AuthManager()
    auth_headers = auth_manager.get_auth_headers()
    
    if not auth_headers:
        st.error("Not authenticated")
        return
    
    # Simple query interface
    st.subheader("💬 Ask a Question")
    
    # Query input
    user_question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What would you like to know?",
        key="simple_query_input"
    )
    
    # Simple settings in an expander
    with st.expander("⚙️ Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Model",
                ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-latest"],
                key="simple_model_choice"
            )
            
        with col2:
            max_results = st.slider(
                "Max Results",
                1, 20, 5,
                key="simple_max_results"
            )
    
    # Query button
    if st.button("🚀 Submit Query", type="primary", use_container_width=True, key="simple_submit"):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                try:
                    # Call the API
                    response = requests.post(
                        "http://localhost:8000/api/v1/query/ask",
                        headers=auth_headers,
                        json={
                            "query": user_question,
                            "max_results": max_results
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Add to chat history
                        st.session_state.simple_chat_messages.append({
                            "role": "user",
                            "content": user_question
                        })
                        st.session_state.simple_chat_messages.append({
                            "role": "assistant", 
                            "content": result.get("answer", "No answer provided"),
                            "metadata": result.get("metadata", {})
                        })
                        
                        st.success("✅ Query completed successfully!")
                        
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")
                        
                except requests.RequestException as e:
                    st.error(f"❌ Network Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question")
    
    # Display chat history
    if st.session_state.simple_chat_messages:
        st.subheader("💭 Conversation History")
        
        # Clear history button
        if st.button("🗑️ Clear History", key="simple_clear_history"):
            st.session_state.simple_chat_messages = []
            st.rerun()
        
        # Display messages
        for i, message in enumerate(st.session_state.simple_chat_messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show metadata if available
                    if message.get("metadata"):
                        with st.expander("📊 Response Details"):
                            metadata = message["metadata"]
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Processing Time", f"{metadata.get('processing_time_ms', 0)}ms")
                                st.metric("Model Used", metadata.get('model_used', 'Unknown'))
                                
                            with col2:
                                st.metric("Tokens Used", metadata.get('tokens_used', 0))
                                st.metric("Confidence", f"{metadata.get('confidence_score', 0)*100:.1f}%")

    # Quick test section
    st.subheader("🧪 Quick Test")
    test_questions = [
        "What is artificial intelligence?",
        "Explain machine learning briefly", 
        "How does natural language processing work?"
    ]
    
    cols = st.columns(len(test_questions))
    for i, question in enumerate(test_questions):
        with cols[i]:
            if st.button(
                f"Test: {question[:20]}...", 
                key=f"test_q_{i}",
                use_container_width=True
            ):
                # Set the question in the text area
                st.session_state.simple_query_input = question
                st.rerun()


if __name__ == "__main__":
    show()