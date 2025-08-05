#!/usr/bin/env python3
"""
Test the improved content processing logic.
"""

import sys
sys.path.insert(0, "src")

from src.episode.processor import EpisodeEmbeddingProcessor, EpisodeProcessingConfig
from src.database.base import DatabaseManager
from src.embedding.manager import EmbeddingManager
from src.core.config import get_config
from sqlalchemy import text

def test_content_processing():
    """Test the improved content processing logic."""
    print("🧪 Testing Improved Content Processing Logic")
    print("=" * 60)
    
    try:
        # Get config and initialize components (without actually using embedding)
        config = get_config()
        db_manager = DatabaseManager(config.database)
        
        # Create a mock embedding manager (we won't actually generate embeddings)
        embedding_manager = None
        
        # Create processor
        processor_config = EpisodeProcessingConfig(
            enable_content_cleaning=True,
            enable_chunking=True,
            chunk_size=1500,
            chunk_overlap=200
        )
        
        # We'll test the methods directly without full processor
        print("📊 Testing Content Processing Methods:")
        
        # Get test content from RDB
        test_episode_id = 239
        with db_manager.get_connection() as conn:
            result = conn.execute(text("""
                SELECT episode_id, episode_title, content 
                FROM episode 
                WHERE episode_id = :episode_id
            """), {'episode_id': test_episode_id})
            episode_data = result.fetchone()
        
        if not episode_data:
            print(f"❌ Test episode {test_episode_id} not found")
            return False
        
        original_content = episode_data.content
        print(f"📖 Original Content (Episode {test_episode_id}):")
        print(f"   Title: {episode_data.episode_title}")
        print(f"   Length: {len(original_content)} chars")
        print(f"   Preview: {original_content[:200]}...")
        
        # Test cleaning function
        from src.episode.processor import EpisodeEmbeddingProcessor
        
        # Create a temporary processor instance to access methods
        temp_processor = EpisodeEmbeddingProcessor(
            database_manager=db_manager,
            embedding_manager=None,  # We won't use this
            config=processor_config
        )
        
        # Test content cleaning
        print(f"\n🧹 Testing Content Cleaning:")
        cleaned_content = temp_processor._clean_content(original_content)
        print(f"   Cleaned Length: {len(cleaned_content)} chars")
        print(f"   Length Change: {len(cleaned_content) - len(original_content):+d} chars")
        print(f"   Preview: {cleaned_content[:200]}...")
        
        # Check for major changes
        if abs(len(cleaned_content) - len(original_content)) < 50:
            print(f"   ✅ Minimal length change (good preservation)")
        else:
            print(f"   ⚠️ Significant length change")
        
        # Test chunking
        print(f"\n✂️ Testing Content Chunking:")
        chunks = temp_processor._split_content_into_chunks(
            cleaned_content, 
            chunk_size=1500, 
            overlap=200
        )
        
        print(f"   Number of chunks: {len(chunks)}")
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        print(f"   Total chunk length: {total_chunk_length} chars")
        print(f"   Length difference: {total_chunk_length - len(cleaned_content):+d} chars")
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i+1}:")
            print(f"     Length: {len(chunk)} chars")
            print(f"     Start: {chunk[:100]}...")
            print(f"     End: ...{chunk[-100:]}")
            
            # Check if chunk exists in original content
            if chunk in cleaned_content:
                print(f"     ✅ Found in cleaned content")
            else:
                # Check word overlap for similarity
                chunk_words = set(chunk.split())
                content_words = set(cleaned_content.split())
                overlap = len(chunk_words & content_words)
                total_chunk_words = len(chunk_words)
                
                if total_chunk_words > 0:
                    overlap_pct = (overlap / total_chunk_words) * 100
                    print(f"     Word overlap: {overlap}/{total_chunk_words} ({overlap_pct:.1f}%)")
                    
                    if overlap_pct >= 90:
                        print(f"     ✅ High word overlap (likely good)")
                    elif overlap_pct >= 70:
                        print(f"     🟡 Moderate word overlap")
                    else:
                        print(f"     ❌ Low word overlap")
        
        # Test reconstruction
        print(f"\n🔄 Testing Content Reconstruction:")
        
        # Simple concatenation (what we currently do)
        simple_reconstruction = "".join(chunks)
        print(f"   Simple concat length: {len(simple_reconstruction)} chars")
        
        # Smart reconstruction (remove overlaps)
        smart_reconstruction = chunks[0] if chunks else ""
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Find overlap between chunks
            max_overlap = min(200, len(prev_chunk), len(curr_chunk))
            best_overlap = 0
            
            for overlap_len in range(max_overlap, 0, -1):
                if prev_chunk[-overlap_len:] == curr_chunk[:overlap_len]:
                    best_overlap = overlap_len
                    break
            
            # Add current chunk without the overlapping part
            smart_reconstruction += curr_chunk[best_overlap:]
        
        print(f"   Smart reconstruction length: {len(smart_reconstruction)} chars")
        print(f"   Original vs Smart diff: {len(smart_reconstruction) - len(cleaned_content):+d} chars")
        
        # Compare with original
        if len(smart_reconstruction) == len(cleaned_content):
            print(f"   ✅ Perfect length match!")
        elif abs(len(smart_reconstruction) - len(cleaned_content)) < 50:
            print(f"   ✅ Very close length match")
        else:
            print(f"   ⚠️ Length mismatch - chunking may need adjustment")
        
        # Character-by-character comparison
        if smart_reconstruction == cleaned_content:
            print(f"   🌟 PERFECT MATCH: Reconstruction identical to original!")
            return True
        else:
            # Find first difference
            min_len = min(len(cleaned_content), len(smart_reconstruction))
            first_diff = None
            for i in range(min_len):
                if cleaned_content[i] != smart_reconstruction[i]:
                    first_diff = i
                    break
            
            if first_diff is not None:
                print(f"   First difference at position {first_diff}")
                print(f"   Context: ...{cleaned_content[max(0,first_diff-30):first_diff+30]}...")
                print(f"   vs:      ...{smart_reconstruction[max(0,first_diff-30):first_diff+30]}...")
            
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    success = test_content_processing()
    print(f"\n{'='*60}")
    if success:
        print("🎉 CONTENT PROCESSING TEST PASSED!")
    else:
        print("⚠️ CONTENT PROCESSING TEST REVEALED ISSUES - but improvements made")
    sys.exit(0 if success else 1)