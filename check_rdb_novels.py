#!/usr/bin/env python3
"""
Check actual RDB data for novels and episode counts.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from sqlalchemy import text

def check_rdb_novels():
    """Check novels and episode counts in actual RDB."""
    print("🔍 Checking RDB Novels and Episode Counts")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize database manager
        db_manager = DatabaseManager(config.database)
        print("✅ Connected to RDB")
        
        # Get total novel count
        with db_manager.get_connection() as conn:
            # Count total novels
            result = conn.execute(text("SELECT COUNT(*) FROM novel"))
            total_novels = result.fetchone()[0]
            print(f"📚 Total novels in RDB: {total_novels}")
            
            # Get episode counts per novel
            result = conn.execute(text("""
                SELECT n.novel_id, n.title, COUNT(e.episode_id) as episode_count
                FROM novel n
                LEFT JOIN episode e ON n.novel_id = e.novel_id  
                GROUP BY n.novel_id, n.title
                ORDER BY episode_count DESC, n.novel_id
            """))
            
            results = result.fetchall()
            
            print(f"\n📊 Episode counts per novel (Top 20):")
            print("-" * 60)
            print(f"{'Novel ID':<10} {'Episodes':<10} {'Title':<40}")
            print("-" * 60)
            
            novels_with_episodes = []
            novels_without_episodes = []
            
            for novel_id, title, episode_count in results[:20]:
                print(f"{novel_id:<10} {episode_count:<10} {title[:40]:<40}")
                if episode_count > 0:
                    novels_with_episodes.append((novel_id, title, episode_count))
                else:
                    novels_without_episodes.append((novel_id, title))
            
            print(f"\n📈 Summary:")
            print(f"- Total novels: {total_novels}")
            print(f"- Novels with episodes: {len(novels_with_episodes)}")
            print(f"- Novels without episodes: {len(novels_without_episodes)}")
            
            if results:
                max_episodes = results[0][2]
                max_novel = results[0]
                print(f"- Novel with most episodes: Novel {max_novel[0]} '{max_novel[1]}' ({max_episodes} episodes)")
            
            # Check if Novel 25 exists and its episode count
            print(f"\n🎯 Novel 25 specific check:")
            result = conn.execute(text("""
                SELECT n.novel_id, n.title, COUNT(e.episode_id) as episode_count
                FROM novel n
                LEFT JOIN episode e ON n.novel_id = e.novel_id  
                WHERE n.novel_id = 25
                GROUP BY n.novel_id, n.title
            """))
            
            novel25_result = result.fetchone()
            if novel25_result:
                novel_id, title, episode_count = novel25_result
                print(f"- Novel 25: '{title}' has {episode_count} episodes")
            else:
                print("- Novel 25: Not found in RDB")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    check_rdb_novels()