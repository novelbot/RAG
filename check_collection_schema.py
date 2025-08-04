#!/usr/bin/env python3
"""
Check Milvus collection schema to understand the actual field structure.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.milvus.client import MilvusClient

def check_collection_schema():
    """Check the actual schema of the episode_embeddings collection."""
    print("🔍 Checking Collection Schema")
    print("=" * 50)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        milvus_client.connect()
        print("✅ Connected to Milvus")
        
        if milvus_client.has_collection("episode_embeddings"):
            collection = milvus_client.get_collection("episode_embeddings")
            schema = collection.schema
            
            print("\n📋 Collection Schema:")
            print("-" * 30)
            
            for field in schema.fields:
                print(f"Field: {field.name}")
                print(f"  Type: {field.dtype}")
                print(f"  Description: {field.description}")
                if hasattr(field, 'params'):
                    print(f"  Params: {field.params}")
                print()
            
            # Also check some sample data
            print("\n📊 Sample Data (first entity):")
            print("-" * 30)
            
            query_results = collection.query(
                expr="",  # Get all
                output_fields=["episode_id", "novel_id", "episode_number", "episode_title", "content_length"],
                limit=1
            )
            
            if query_results:
                sample = query_results[0]
                for key, value in sample.items():
                    print(f"{key}: {value}")
                    
        else:
            print("❌ Collection 'episode_embeddings' does not exist")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_collection_schema()