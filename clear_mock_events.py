#!/usr/bin/env python3
"""
Clear mock events from the database to show only real user activities.
"""

import asyncio
import sqlite3
from pathlib import Path

async def clear_mock_events():
    """Clear existing mock events from the system_events table."""
    db_path = Path("metrics.db")
    
    if not db_path.exists():
        print("❌ No metrics database found. Run the server first to create it.")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Clear all existing system events
            cursor = conn.execute("DELETE FROM system_events")
            deleted_count = cursor.rowcount
            conn.commit()
            
            print(f"✅ Cleared {deleted_count} mock events from the database")
            print("Now the Recent Activity section will only show real user activities!")
            
    except Exception as e:
        print(f"❌ Failed to clear mock events: {e}")

if __name__ == "__main__":
    asyncio.run(clear_mock_events())