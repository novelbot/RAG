#!/usr/bin/env python3
"""
Check episode count in RDB directly using episode table access.
"""

import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.database.base import DatabaseManager
from sqlalchemy import text

def check_rdb_episode_count():
    """Check episode counts in RDB using episode table."""
    print("🔍 Checking RDB Episode Counts")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        
        # Initialize database manager
        db_manager = DatabaseManager(config.database)
        print("✅ Connected to RDB")
        
        with db_manager.get_connection() as conn:
            # Try to access episode table first
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM episode"))
                total_episodes = result.fetchone()[0]
                print(f"📊 Total episodes in RDB: {total_episodes}")
                
                # Get episode counts per novel_id
                result = conn.execute(text("""
                    SELECT novel_id, COUNT(*) as episode_count
                    FROM episode 
                    GROUP BY novel_id
                    ORDER BY episode_count DESC, novel_id
                """))
                
                results = result.fetchall()
                
                print(f"\n📚 Novel episode counts (Top 20):")
                print("-" * 40)
                print(f"{'Novel ID':<10} {'Episodes':<10}")
                print("-" * 40)
                
                novel_count = len(results)
                for i, (novel_id, episode_count) in enumerate(results[:20]):
                    print(f"{novel_id:<10} {episode_count:<10}")
                
                print(f"\n📈 Summary:")
                print(f"- Total novels with episodes: {novel_count}")
                print(f"- Total episodes: {total_episodes}")
                
                if results:
                    max_episodes = results[0][1]
                    max_novel_id = results[0][0]
                    print(f"- Novel with most episodes: Novel {max_novel_id} ({max_episodes} episodes)")
                
                # Check Novel 25 specifically
                result = conn.execute(text("SELECT COUNT(*) FROM episode WHERE novel_id = 25"))
                novel25_count = result.fetchone()[0]
                print(f"- Novel 25 episodes: {novel25_count}")
                
                # Get distinct novel_ids to count total novels
                result = conn.execute(text("SELECT COUNT(DISTINCT novel_id) FROM episode"))
                distinct_novels = result.fetchone()[0]
                print(f"- Distinct novels in episode table: {distinct_novels}")
                
            except Exception as e:
                print(f"❌ Episode table access error: {e}")
                
                # Try alternative approach - check what tables we can access
                print("\n🔍 Checking available tables...")
                try:
                    result = conn.execute(text("SHOW TABLES"))
                    tables = result.fetchall()
                    print("Available tables:")
                    for table in tables:
                        print(f"  - {table[0]}")
                except Exception as show_error:
                    print(f"❌ Cannot show tables: {show_error}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.close()

if __name__ == "__main__":
    check_rdb_episode_count()