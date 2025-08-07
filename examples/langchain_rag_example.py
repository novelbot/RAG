"""
Comprehensive Example: Using LangChain-based RAG System

This example demonstrates how to use the refactored LangChain implementations
for building a complete RAG (Retrieval-Augmented Generation) pipeline.
"""

import asyncio
import os
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports - adjust path as needed
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding.langchain_embeddings import (
    LangChainEmbeddingProvider, LangChainEmbeddingConfig
)
from src.vector_stores.langchain_milvus import (
    LangChainMilvusVectorStore, LangChainMilvusConfig
)
from src.rag.langchain_rag import (
    LangChainRAG, LangChainRAGConfig, RAGStrategy
)
from src.llm.langchain_providers import (
    LangChainLLMManager, LangChainLLMConfig,
    RAGLangChainProvider
)


class LangChainRAGExample:
    """Example implementation of LangChain-based RAG system."""
    
    def __init__(self):
        """Initialize the example."""
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.rag_system = None
    
    async def setup(self):
        """Setup all components."""
        print("🚀 Setting up LangChain RAG System...")
        
        # 1. Initialize LLM
        self.llm = self._setup_llm()
        print("✅ LLM initialized")
        
        # 2. Initialize Embeddings
        self.embeddings = self._setup_embeddings()
        print("✅ Embeddings initialized")
        
        # 3. Initialize Vector Store
        self.vector_store = await self._setup_vector_store()
        print("✅ Vector store initialized")
        
        # 4. Initialize RAG System
        self.rag_system = self._setup_rag_system()
        print("✅ RAG system initialized")
    
    def _setup_llm(self):
        """Setup LangChain LLM."""
        # Option 1: Use standard LangChain LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        
        # Option 2: Use custom LangChainLLMManager for more flexibility
        # config = LangChainLLMConfig(
        #     provider="openai",
        #     model="gpt-3.5-turbo",
        #     temperature=0.7,
        #     streaming=True
        # )
        # manager = LangChainLLMManager(config)
        # llm = manager.llm_client
        
        return llm
    
    def _setup_embeddings(self):
        """Setup LangChain embeddings."""
        # Option 1: Use standard LangChain embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Option 2: Use custom LangChainEmbeddingProvider
        # config = LangChainEmbeddingConfig(
        #     provider="openai",
        #     model="text-embedding-3-small",
        #     batch_size=100
        # )
        # provider = LangChainEmbeddingProvider(config)
        # embeddings = provider.embeddings_client
        
        return embeddings
    
    async def _setup_vector_store(self):
        """Setup Milvus vector store."""
        config = LangChainMilvusConfig(
            host="localhost",
            port=19530,
            collection_name="langchain_example",
            dimension=1536,  # OpenAI embedding dimension
            metric_type="L2"
        )
        
        vector_store = LangChainMilvusVectorStore(
            embedding_function=self.embeddings,
            config=config
        )
        
        return vector_store
    
    def _setup_rag_system(self):
        """Setup RAG system with various strategies."""
        # Get retriever from vector store
        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7
            }
        )
        
        # Configure RAG system
        config = LangChainRAGConfig(
            strategy=RAGStrategy.CONVERSATIONAL,  # Try different strategies
            retrieval_k=5,
            retrieval_score_threshold=0.7,
            use_mmr=True,
            mmr_lambda=0.5,
            temperature=0.7,
            max_tokens=1000,
            streaming=True,
            return_source_documents=True,
            use_memory=True,
            system_prompt="""You are a helpful AI assistant specialized in answering questions 
            based on provided documents. Always cite your sources when possible."""
        )
        
        rag_system = LangChainRAG(
            llm=self.llm,
            retriever=retriever,
            config=config
        )
        
        return rag_system
    
    async def ingest_documents(self, documents: List[str]):
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of document texts
        """
        print("\n📚 Ingesting documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        
        # Process each document
        all_chunks = []
        for i, doc_text in enumerate(documents):
            # Create document
            doc = Document(
                page_content=doc_text,
                metadata={
                    "source": f"document_{i}",
                    "doc_id": i
                }
            )
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        # Add to vector store
        ids = await self.vector_store.aadd_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} chunks from {len(documents)} documents")
        
        return ids
    
    async def query(self, question: str, chat_history: List = None):
        """
        Query the RAG system.
        
        Args:
            question: User question
            chat_history: Optional conversation history
        """
        print(f"\n❓ Query: {question}")
        
        # Query RAG system
        result = await self.rag_system.aquery(
            question=question,
            chat_history=chat_history
        )
        
        # Display results
        print(f"\n💡 Answer: {result['answer']}")
        
        if result.get('sources'):
            print("\n📖 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  Source {i}:")
                print(f"    Content: {source['content'][:200]}...")
                print(f"    Metadata: {source['metadata']}")
        
        print(f"\n⏱️ Response time: {result['response_time']:.2f}s")
        print(f"📊 Strategy used: {result['strategy']}")
        
        return result
    
    async def demo_different_strategies(self):
        """Demonstrate different RAG strategies."""
        strategies = [
            RAGStrategy.SIMPLE,
            RAGStrategy.MULTI_QUERY,
            RAGStrategy.CONTEXTUAL_COMPRESSION,
            RAGStrategy.CONVERSATIONAL
        ]
        
        question = "What are the main topics discussed in the documents?"
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Testing strategy: {strategy.value}")
            print('='*60)
            
            # Reconfigure RAG system with new strategy
            self.rag_system.config.strategy = strategy
            self.rag_system.retriever = self.rag_system._setup_retriever()
            
            # Query
            await self.query(question)
    
    async def demo_conversation(self):
        """Demonstrate conversational RAG."""
        print("\n" + "="*60)
        print("CONVERSATIONAL RAG DEMO")
        print("="*60)
        
        # Set to conversational strategy
        self.rag_system.config.strategy = RAGStrategy.CONVERSATIONAL
        self.rag_system.retriever = self.rag_system._setup_retriever()
        
        # Conversation flow
        questions = [
            "What is machine learning?",
            "Can you give me an example of that?",
            "How does it compare to traditional programming?"
        ]
        
        chat_history = []
        
        for q in questions:
            result = await self.query(q, chat_history)
            
            # Update chat history
            chat_history.append((q, result['answer']))
        
        # Show conversation memory
        print("\n📝 Conversation Memory:")
        memory_messages = self.rag_system.get_memory_messages()
        for msg in memory_messages:
            print(f"  {msg['role']}: {msg['content'][:100]}...")


async def main():
    """Main function to run the example."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        return
    
    # Create example instance
    example = LangChainRAGExample()
    
    try:
        # Setup
        await example.setup()
        
        # Sample documents
        sample_documents = [
            """Machine learning is a subset of artificial intelligence (AI) that provides 
            systems the ability to automatically learn and improve from experience without 
            being explicitly programmed. Machine learning focuses on the development of 
            computer programs that can access data and use it to learn for themselves.""",
            
            """Deep learning is a subset of machine learning that uses neural networks 
            with multiple layers (deep neural networks) to progressively extract higher 
            level features from raw input. For example, in image processing, lower layers 
            may identify edges, while higher layers may identify human-relevant concepts.""",
            
            """Natural Language Processing (NLP) is a branch of artificial intelligence 
            that helps computers understand, interpret and manipulate human language. 
            NLP draws from many disciplines, including computer science and computational 
            linguistics, to bridge the gap between human communication and computer understanding."""
        ]
        
        # Ingest documents
        await example.ingest_documents(sample_documents)
        
        # Example 1: Simple query
        print("\n" + "="*60)
        print("EXAMPLE 1: SIMPLE QUERY")
        print("="*60)
        await example.query("What is machine learning?")
        
        # Example 2: Complex query
        print("\n" + "="*60)
        print("EXAMPLE 2: COMPLEX QUERY")
        print("="*60)
        await example.query(
            "Compare and contrast machine learning, deep learning, and NLP. "
            "How are they related?"
        )
        
        # Example 3: Conversational RAG
        await example.demo_conversation()
        
        # Example 4: Different strategies
        # await example.demo_different_strategies()
        
        # Clean up
        print("\n🧹 Cleaning up...")
        # Optionally drop collection
        # example.vector_store.drop_collection()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())