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
            k_value = st.slider("Number of retrieved documents (k)", 1, 20, 5)
            # Get available LLM providers from config
            available_providers = list(config.get_enabled_llm_providers().keys())
            if not available_providers:
                available_providers = ["openai", "anthropic", "google", "ollama"]
            
            llm_provider = st.selectbox(
                "LLM Provider",
                available_providers
            )
        
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.1)
            model_options = {
                "openai": ["gpt-4", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-5-sonnet-latest", "claude-3-haiku"],
                "google": ["gemini-2.0-flash-001", "gemini-1.5-pro"],
                "ollama": ["llama3.2", "gemma2"]
            }
            model = st.selectbox("Model", model_options.get(llm_provider, ["gpt-4"]))
    
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
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max tokens", 100, 4000, 1000)
    
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
                ["gpt-4", "gpt-3.5-turbo", "claude-3-5-sonnet-latest", "gemini-2.0-flash-001"]
            )
        
        with col2:
            chat_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            chat_max_tokens = st.number_input("Max tokens", 100, 4000, 1000)
    
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
            ["gpt-4", "claude-3-5-sonnet-latest", "gemini-2.0-flash-001", "llama3.2"]
        )
        context = st.text_area("Context (optional)", height=100)
    
    with col2:
        single_temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="single_temp")
        response_format = st.selectbox("Response Format", ["text", "markdown", "json"])
    
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
        ensemble_size = st.slider("Ensemble size", 2, 5, 3)
        consensus_threshold = st.slider("Consensus threshold", 0.5, 1.0, 0.7, 0.1)
    
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
            # Mock RAG response for demo
            response = {
                "answer": "Based on the retrieved documents, the company's Q1 2024 revenue was $2.3 million, representing a 15% increase compared to Q4 2023. This growth was primarily driven by increased product sales and new client acquisitions.",
                "sources": [
                    {
                        "document": "quarterly_report_q1_2024.pdf",
                        "relevance_score": 0.95,
                        "content_preview": "Q1 2024 revenue: $2.3M, up 15% from previous quarter..."
                    },
                    {
                        "document": "financial_summary_2024.xlsx",
                        "relevance_score": 0.87,
                        "content_preview": "Revenue breakdown by segment shows consistent growth..."
                    }
                ],
                "metadata": {
                    "query_time": "2.3 seconds",
                    "documents_retrieved": config["k"],
                    "model_used": f"{config['llm_provider']}/{config['model']}"
                }
            }
            
            # Display response
            st.success("✅ Query completed successfully!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Answer")
                st.write(response["answer"])
                
                st.subheader("📚 Sources")
                for i, source in enumerate(response["sources"], 1):
                    with st.expander(f"Source {i}: {source['document']} (Relevance: {source['relevance_score']:.2f})"):
                        st.write(source["content_preview"])
            
            with col2:
                st.subheader("📊 Query Metadata")
                st.json(response["metadata"])
            
            # Save to history
            save_query_to_history("rag_queries", query, response, config)
            
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")

def execute_chat_query(api_client, messages, config):
    """Execute chat query"""
    try:
        # Mock chat response for demo
        user_message = messages[-1]["content"]
        
        # Simple mock responses based on content
        if "weather" in user_message.lower():
            response_content = "I don't have access to real-time weather data, but I can help you understand weather patterns or suggest reliable weather sources."
        elif "time" in user_message.lower():
            response_content = f"The current time is approximately {datetime.now().strftime('%H:%M')}. However, I recommend checking your system clock for the most accurate time."
        else:
            response_content = f"I understand you're asking about: '{user_message}'. While I can't process this through the actual API in this demo, I can help you formulate better queries or discuss the topic in general terms."
        
        return {
            "content": response_content,
            "metadata": {
                "model": config["model"],
                "temperature": config["temperature"],
                "response_time": "1.2 seconds"
            }
        }
        
    except Exception as e:
        st.error(f"❌ Chat failed: {str(e)}")
        return None

def execute_single_llm_query(api_client, query, config):
    """Execute single LLM query"""
    with st.spinner("Generating response..."):
        try:
            # Mock single LLM response
            response = {
                "response": f"This is a mock response to your query: '{query}'. In a real implementation, this would be generated by {config['model']} with temperature {config['temperature']}.",
                "metadata": {
                    "model": config["model"],
                    "temperature": config["temperature"],
                    "response_time": "1.8 seconds",
                    "tokens_used": 87
                }
            }
            
            st.success("✅ Response generated!")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📝 Response")
                if config["response_format"] == "markdown":
                    st.markdown(response["response"])
                elif config["response_format"] == "json":
                    try:
                        st.json(json.loads(response["response"]))
                    except:
                        st.code(response["response"], language="json")
                else:
                    st.write(response["response"])
            
            with col2:
                st.subheader("📊 Metadata")
                st.json(response["metadata"])
                
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")

def execute_ensemble_query(api_client, query, config):
    """Execute ensemble LLM query"""
    with st.spinner("Generating ensemble responses..."):
        try:
            # Mock ensemble response
            response = {
                "best_response": f"Ensemble response to: '{query}'. This combines insights from {config['ensemble_size']} different models.",
                "all_responses": [
                    {"model": "gpt-4", "response": "GPT-4's perspective on the query...", "quality_score": 0.92},
                    {"model": "claude-3", "response": "Claude's analysis of the topic...", "quality_score": 0.89},
                    {"model": "gemini", "response": "Gemini's interpretation...", "quality_score": 0.86}
                ],
                "consensus_score": 0.78,
                "metadata": {
                    "ensemble_size": config["ensemble_size"],
                    "consensus_threshold": config["consensus_threshold"],
                    "processing_time": "12.5 seconds",
                    "selection_method": "quality_score"
                }
            }
            
            st.success("✅ Ensemble response generated!")
            
            # Best response
            st.subheader("🏆 Best Response")
            st.write(response["best_response"])
            
            # All responses comparison
            st.subheader("📊 Response Comparison")
            
            for i, resp in enumerate(response["all_responses"], 1):
                with st.expander(f"Response {i}: {resp['model']} (Quality: {resp['quality_score']:.2f})"):
                    st.write(resp["response"])
            
            # Metadata
            st.subheader("📈 Ensemble Metadata")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Consensus Score", f"{response['consensus_score']:.2f}")
                st.metric("Processing Time", response["metadata"]["processing_time"])
            
            with col2:
                st.metric("Ensemble Size", response["metadata"]["ensemble_size"])
                st.write(f"**Selection Method:** {response['metadata']['selection_method']}")
                
        except Exception as e:
            st.error(f"❌ Ensemble generation failed: {str(e)}")

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