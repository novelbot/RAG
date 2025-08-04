#!/usr/bin/env python3
"""
Process real episode data from RDB and store embeddings in Milvus.
Uses existing episode embedding functionality with .env configuration.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import get_config
from src.core.logging import get_logger
from src.database.base import DatabaseManager
from src.embedding.factory import get_embedding_manager
from src.milvus.client import MilvusClient
from src.episode.manager import EpisodeRAGManager, EpisodeRAGConfig

logger = get_logger(__name__)

async def main():
    """Main processing function."""
    try:
        # Load configuration from .env
        config = get_config()
        logger.info("Loaded configuration from .env")
        
        # Initialize database manager with database config
        db_manager = DatabaseManager(config.database)
        logger.info(f"Initialized database connection to {config.database.host}:{config.database.port}/{config.database.database}")
        
        # Initialize embedding manager using factory
        embedding_manager = get_embedding_manager([config.embedding])
        logger.info(f"Initialized embedding manager with {config.embedding.provider}:{config.embedding.model}")
        
        # Test embedding to get dimension
        logger.info("Testing embedding generation to determine vector dimension...")
        from src.embedding.base import EmbeddingRequest
        test_request = EmbeddingRequest(
            input=["test text"],
            model=config.embedding.model,
            encoding_format="float"
        )
        test_response = embedding_manager.generate_embeddings(test_request)
        vector_dimension = len(test_response.embeddings[0])
        logger.info(f"Detected vector dimension: {vector_dimension}")
        
        # Initialize Milvus client
        milvus_client = MilvusClient(config.milvus)
        logger.info(f"Initialized Milvus client: {config.milvus.host}:{config.milvus.port}")
        
        # Connect to Milvus
        logger.info("Connecting to Milvus server...")
        milvus_client.connect()
        logger.info("Successfully connected to Milvus server")
        
        # Create episode RAG configuration with .env settings
        episode_config = EpisodeRAGConfig(
            processing_batch_size=50,  # Smaller batches for stability
            embedding_model=config.embedding.model,
            enable_content_cleaning=True,
            collection_name="episode_embeddings",
            vector_dimension=vector_dimension,
            index_params={"nlist": 1024},
            default_search_limit=10,
            max_search_limit=100
        )
        
        # Initialize Episode RAG Manager
        logger.info("Initializing Episode RAG Manager...")
        episode_manager = EpisodeRAGManager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config
        )
        
        # Setup collection
        logger.info("Setting up Milvus collection...")
        setup_result = await episode_manager.setup_collection(drop_existing=True)
        logger.info(f"Collection setup result: {setup_result}")
        
        # Get list of novels from database
        logger.info("Getting list of novels from database...")
        with db_manager.get_connection() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT DISTINCT novel_id FROM episode ORDER BY novel_id"))
            novel_ids = [row[0] for row in result.fetchall()]
        
        logger.info(f"Found {len(novel_ids)} novels to process")
        
        # Process novels in batches
        processed_novels = 0
        total_episodes_processed = 0
        
        for i, novel_id in enumerate(novel_ids):
            try:
                logger.info(f"Processing novel {novel_id} ({i+1}/{len(novel_ids)})...")
                
                # Process the novel
                result = await episode_manager.process_novel(novel_id)
                
                processed_novels += 1
                total_episodes_processed += result.get('episodes_processed', 0)
                
                logger.info(f"Novel {novel_id} completed: {result.get('episodes_processed', 0)} episodes processed")
                
                # Log progress every 10 novels
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(novel_ids)} novels, {total_episodes_processed} total episodes")
                
            except Exception as e:
                logger.error(f"Failed to process novel {novel_id}: {e}")
                continue
        
        # Get final statistics
        stats = episode_manager.get_manager_stats()
        logger.info("=== Processing Complete ===")
        logger.info(f"Novels processed: {processed_novels}")
        logger.info(f"Total episodes processed: {total_episodes_processed}")
        logger.info(f"Manager statistics: {stats}")
        
        # Perform health check
        health = episode_manager.health_check()
        logger.info(f"System health check: {health['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup
        try:
            if 'episode_manager' in locals():
                episode_manager.close()
            if 'db_manager' in locals():
                db_manager.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("✅ Episode embedding processing completed successfully!")
    else:
        print("❌ Episode embedding processing failed!")
        sys.exit(1)