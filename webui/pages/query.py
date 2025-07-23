"""
Query interface page for RAG Server Web UI
Handles RAG queries, chat interactions, and result display
"""

import streamlit as st
import json
from datetime import datetime
from webui.api_client import get_api_client
from webui.auth import require_auth
from webui.config import config
import time

@require_auth
def show():
    """Display the query interface page"""
    st.title("🔍 Query Interface")
    
    api_client = get_api_client()
    
    # Create tabs for different query modes
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 RAG Query", "💬 Chat", "🎯 Single LLM", "🎪 Ensemble"])
    
    with tab1:
        show_rag_query(api_client)
    
    with tab2:
        show_chat_interface(api_client)
    
    with tab3:
        show_single_llm(api_client)
    
    with tab4:
        show_ensemble_llm(api_client)

def show_rag_query(api_client):
    """Display RAG query interface"""
    st.subheader("🧠 RAG Query")
    st.write("Query the knowledge base using Retrieval-Augmented Generation")
    
    # Query configuration
    with st.expander("⚙️ Query Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            k_value = st.slider("Number of retrieved documents (k)", 1, 20, 5, key="rag_k_value")
            # Get available LLM providers from config
            available_providers = list(config.get_enabled_llm_providers().keys())
            if not available_providers:
                available_providers = ["openai", "anthropic", "google", "ollama"]
            
            llm_provider = st.selectbox(
                "LLM Provider",
                available_providers,
                key="rag_llm_provider"
            )
        
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.1, key="rag_similarity_threshold")
            model_options = {
                "openai": ["gpt-4", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-5-sonnet-latest", "claude-3-haiku"],
                "google": ["gemini-2.0-flash-001", "gemini-1.5-pro"],
                "ollama": ["llama3.2", "gemma2"]
            }
            model = st.selectbox("Model", model_options.get(llm_provider, ["gpt-4"]), key="rag_model")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What is the company's revenue for Q1 2024?",
        height=100
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_metadata = st.checkbox("Include source metadata", value=True)
            enable_reranking = st.checkbox("Enable result reranking", value=False)
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="rag_temperature")
            max_tokens = st.number_input("Max tokens", 100, 4000, 1000, key="rag_max_tokens")
    
    # Query button
    if st.button("🚀 Query", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a question")
            return
        
        execute_rag_query(api_client, query, {
            "k": k_value,
            "llm_provider": llm_provider,
            "model": model,
            "similarity_threshold": similarity_threshold,
            "include_metadata": include_metadata,
            "enable_reranking": enable_reranking,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    
    # Query history
    show_query_history("rag_queries")

def show_chat_interface(api_client):
    """Display chat interface"""
    st.subheader("💬 Chat with LLM")
    st.write("Direct conversation with language models")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Chat configuration
    with st.expander("⚙️ Chat Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            chat_model = st.selectbox(
                "Model",
                ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-latest", "gemini-2.0-flash-001"],
                key="chat_model"
            )
        
        with col2:
            chat_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="chat_temperature")
            chat_max_tokens = st.number_input("Max tokens", 100, 4000, 1000, key="chat_max_tokens")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("metadata"):
                    with st.expander("📊 Message Metadata"):
                        st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
        
        # Get AI response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = execute_chat_query(api_client, st.session_state.chat_messages, {
                        "model": chat_model,
                        "temperature": chat_temperature,
                        "max_tokens": chat_max_tokens
                    })
                    
                    if response:
                        st.write(response["content"])
                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": response["content"],
                            "metadata": response.get("metadata", {})
                        })
    
    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            save_chat_history(st.session_state.chat_messages)

def show_single_llm(api_client):
    """Display single LLM interface"""
    st.subheader("🎯 Single LLM Generation")
    st.write("Fast response generation using a single language model")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        single_model = st.selectbox(
            "Model",
            ["gpt-4", "claude-3-5-sonnet-latest", "gemini-2.0-flash-001", "llama3.2"],
            key="single_llm_model"
        )
        context = st.text_area("Context (optional)", height=100, key="single_context")
    
    with col2:
        single_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="single_temp")
        response_format = st.selectbox("Response Format", ["text", "markdown", "json"], key="single_response_format")
    
    # Query input
    single_query = st.text_area(
        "Enter your prompt:",
        placeholder="Explain quantum computing in simple terms",
        height=100,
        key="single_query"
    )
    
    # Generate button
    if st.button("⚡ Generate", type="primary", use_container_width=True, key="single_generate"):
        if not single_query.strip():
            st.error("Please enter a prompt")
            return
        
        execute_single_llm_query(api_client, single_query, {
            "model": single_model,
            "context": context,
            "temperature": single_temperature,
            "response_format": response_format
        })

def show_ensemble_llm(api_client):
    """Display ensemble LLM interface"""
    st.subheader("🎪 Ensemble LLM Generation")
    st.write("High-quality responses using multiple language models")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        ensemble_size = st.slider("Ensemble size", 2, 5, 3, key="ensemble_size")
        consensus_threshold = st.slider("Consensus threshold", 0.5, 1.0, 0.7, 0.1, key="ensemble_consensus_threshold")
    
    with col2:
        parallel_generation = st.checkbox("Parallel generation", value=True)
        enable_evaluation = st.checkbox("Enable response evaluation", value=True)
    
    # Evaluation metrics
    if enable_evaluation:
        evaluation_metrics = st.multiselect(
            "Evaluation metrics",
            ["relevance", "accuracy", "completeness", "clarity", "coherence"],
            default=["relevance", "accuracy", "completeness"]
        )
    
    # Query input
    ensemble_query = st.text_area(
        "Enter your prompt:",
        placeholder="Analyze the impact of artificial intelligence on modern business",
        height=100,
        key="ensemble_query"
    )
    
    # Generate button
    if st.button("🎭 Generate Ensemble", type="primary", use_container_width=True, key="ensemble_generate"):
        if not ensemble_query.strip():
            st.error("Please enter a prompt")
            return
        
        execute_ensemble_query(api_client, ensemble_query, {
            "ensemble_size": ensemble_size,
            "consensus_threshold": consensus_threshold,
            "parallel_generation": parallel_generation,
            "enable_evaluation": enable_evaluation,
            "evaluation_metrics": evaluation_metrics if enable_evaluation else []
        })

def execute_rag_query(api_client, query, config):
    """Execute RAG query and display results"""
    with st.spinner("Searching knowledge base..."):
        try:
            # Call actual RAG API
            response = api_client.query_rag(
                query=query,
                k=config["k"],
                llm_provider=config["llm_provider"],
                model=config["model"]
            )
            
            # Display response
            st.success("✅ Query completed successfully!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Answer")
                st.write(response.get("answer", "No answer provided"))
                
                st.subheader("📚 Sources")
                sources = response.get("sources", [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        relevance_score = source.get("relevance_score", 0.0)
                        document_title = source.get("title", source.get("document_id", f"Source {i}"))
                        excerpt = source.get("excerpt", source.get("content_preview", "No preview available"))
                        
                        with st.expander(f"Source {i}: {document_title} (Relevance: {relevance_score:.2f})"):
                            st.write(excerpt)
                else:
                    st.info("No sources found for this query")
            
            with col2:
                st.subheader("📊 Query Metadata")
                metadata = response.get("metadata", {})
                display_metadata = {
                    "processing_time_ms": metadata.get("processing_time_ms", 0),
                    "model_used": metadata.get("model_used", f"{config['llm_provider']}/{config['model']}"),
                    "tokens_used": metadata.get("tokens_used", 0),
                    "confidence_score": metadata.get("confidence_score", 0.0)
                }
                st.json(display_metadata)
            
            # Save to history
            save_query_to_history("rag_queries", query, response, config)
            
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")
            st.error("Please check if the RAG server is running and properly configured.")

def execute_chat_query(api_client, messages, config):
    """Execute chat query"""
    try:
        # Convert messages to API format
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Call actual chat API
        response = api_client.chat_llm(
            messages=api_messages,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        if response:
            return {
                "content": response.get("response", response.get("content", "No response received")),
                "metadata": response.get("metadata", {
                    "model": config["model"],
                    "temperature": config["temperature"],
                    "response_time": "unknown"
                })
            }
        else:
            return None
        
    except Exception as e:
        st.error(f"❌ Chat failed: {str(e)}")
        return None

def execute_single_llm_query(api_client, query, config):
    """Execute single LLM query"""
    with st.spinner("Generating response..."):
        try:
            # Call actual single LLM API
            response = api_client.generate_single_llm(
                query=query,
                context=config.get("context", ""),
                model=config["model"],
                temperature=config["temperature"]
            )
            
            st.success("✅ Response generated!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📝 Response")
                response_content = response.get("response", response.get("content", "No response received"))
                
                if config["response_format"] == "markdown":
                    st.markdown(response_content)
                elif config["response_format"] == "json":
                    try:
                        st.json(json.loads(response_content))
                    except:
                        st.code(response_content, language="json")
                else:
                    st.write(response_content)
            
            with col2:
                st.subheader("📊 Metadata")
                metadata = response.get("metadata", {
                    "model": config["model"],
                    "temperature": config["temperature"],
                    "response_time": "unknown",
                    "tokens_used": 0
                })
                st.json(metadata)
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            st.error("Please check if the RAG server is running and properly configured.")

def execute_ensemble_query(api_client, query, config):
    """Execute ensemble LLM query"""
    with st.spinner("Generating ensemble responses..."):
        try:
            # Call actual ensemble LLM API
            response = api_client.generate_ensemble_llm(
                query=query,
                ensemble_size=config["ensemble_size"],
                consensus_threshold=config["consensus_threshold"]
            )
            
            st.success("✅ Ensemble response generated!")
            
            # Best response
            st.subheader("🏆 Best Response")
            best_response = response.get("best_response", response.get("response", "No response received"))
            st.write(best_response)
            
            # All responses comparison (if available)
            all_responses = response.get("all_responses", [])
            if all_responses:
                st.subheader("📊 Response Comparison")
                
                for i, resp in enumerate(all_responses, 1):
                    model_name = resp.get("model", f"Model {i}")
                    quality_score = resp.get("quality_score", 0.0)
                    response_text = resp.get("response", "No response")
                    
                    with st.expander(f"Response {i}: {model_name} (Quality: {quality_score:.2f})"):
                        st.write(response_text)
            
            # Metadata
            st.subheader("📈 Ensemble Metadata")
            col1, col2 = st.columns(2)
            
            metadata = response.get("metadata", {})
            consensus_score = response.get("consensus_score", metadata.get("consensus_score", 0.0))
            processing_time = metadata.get("processing_time", "unknown")
            ensemble_size = metadata.get("ensemble_size", config["ensemble_size"])
            selection_method = metadata.get("selection_method", "quality_score")
            
            with col1:
                st.metric("Consensus Score", f"{consensus_score:.2f}")
                st.metric("Processing Time", processing_time)
            
            with col2:
                st.metric("Ensemble Size", ensemble_size)
                st.write(f"**Selection Method:** {selection_method}")
                
        except Exception as e:
            st.error(f"❌ Ensemble generation failed: {str(e)}")
            st.error("Please check if the RAG server is running and properly configured.")

def show_query_history(history_key):
    """Display query history"""
    if f"{history_key}_history" not in st.session_state:
        st.session_state[f"{history_key}_history"] = []
    
    history = st.session_state[f"{history_key}_history"]
    
    if history:
        st.subheader("📜 Query History")
        
        with st.expander(f"View History ({len(history)} queries)"):
            for i, entry in enumerate(reversed(history[-10:])):  # Show last 10
                st.write(f"**{i+1}.** {entry['query']}")
                st.write(f"*{entry['timestamp']}*")
                if st.button(f"Rerun", key=f"rerun_{history_key}_{i}"):
                    st.text_area("Query", value=entry['query'], key=f"rerun_query_{i}")

def save_query_to_history(history_key, query, response, config):
    """Save query to history"""
    if f"{history_key}_history" not in st.session_state:
        st.session_state[f"{history_key}_history"] = []
    
    st.session_state[f"{history_key}_history"].append({
        "query": query,
        "response": response,
        "config": config,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def save_chat_history(messages):
    """Save chat history"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    # In a real implementation, this would save to file or database
    st.success(f"Chat history saved as {filename}")
    st.download_button(
        "💾 Download Chat History",
        data=json.dumps(messages, indent=2),
        file_name=filename,
        mime="application/json"
    )