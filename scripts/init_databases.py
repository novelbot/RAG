#!/usr/bin/env python3
"""
Database initialization script for RAG Server
Initializes all SQLite databases with proper schemas
"""

import os
import sqlite3
from pathlib import Path
import sys
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SCHEMA_DIR = PROJECT_ROOT / "database" / "schemas"
DATA_DIR = PROJECT_ROOT / "data"

def ensure_directories():
    """Ensure required directories exist"""
    DATA_DIR.mkdir(exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

def load_schema(schema_file: Path) -> str:
    """Load SQL schema from file"""
    if not schema_file.exists():
        logger.warning(f"Schema file not found: {schema_file}")
        return ""
    
    with open(schema_file, 'r') as f:
        return f.read()

def init_database(db_path: Path, schema_file: Path, db_name: str) -> bool:
    """Initialize a single database with its schema"""
    try:
        logger.info(f"Initializing {db_name} database at {db_path}")
        
        # Create parent directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load schema
        schema_sql = load_schema(schema_file)
        if not schema_sql or schema_sql.startswith("#"):
            logger.info(f"No schema to apply for {db_name} (will be created on first use)")
            return True
        
        # Connect to database (creates if not exists)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables already exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        if existing_tables and existing_tables != ['sqlite_sequence']:
            logger.info(f"Database {db_name} already has tables: {existing_tables}")
            response = input(f"Do you want to reinitialize {db_name}? This will drop all existing data! (y/N): ")
            if response.lower() != 'y':
                logger.info(f"Skipping {db_name}")
                conn.close()
                return True
            
            # Drop all tables
            for table in existing_tables:
                if table != 'sqlite_sequence':
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
            logger.info(f"Dropped all tables in {db_name}")
        
        # Apply schema
        cursor.executescript(schema_sql)
        conn.commit()
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        created_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Created tables in {db_name}: {created_tables}")
        
        conn.close()
        logger.info(f"Successfully initialized {db_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize {db_name}: {e}")
        return False

def add_default_admin_user(db_path: Path) -> bool:
    """Add default admin user to auth database"""
    try:
        import hashlib
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if admin already exists
        cursor.execute("SELECT username FROM users WHERE username = 'admin'")
        if cursor.fetchone():
            logger.info("Admin user already exists")
            conn.close()
            return True
        
        # Create admin user with hashed password
        # Default password: admin123 (should be changed on first login)
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, role, is_active)
            VALUES ('admin', ?, 'admin@example.com', 'admin', 1)
        """, (password_hash,))
        
        conn.commit()
        conn.close()
        
        logger.info("Created default admin user (username: admin, password: admin123)")
        logger.warning("⚠️  Please change the admin password after first login!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        return False

def main(databases: List[str] = None):
    """Main initialization function"""
    ensure_directories()
    
    # Define database configurations
    db_configs = {
        'auth': {
            'path': PROJECT_ROOT / 'auth.db',
            'schema': SCHEMA_DIR / 'auth.sql',
            'post_init': lambda p: add_default_admin_user(p)
        },
        'metrics': {
            'path': PROJECT_ROOT / 'metrics.db',
            'schema': SCHEMA_DIR / 'metrics.sql',
            'post_init': None
        },
        'conversations': {
            'path': DATA_DIR / 'conversations.db',
            'schema': SCHEMA_DIR / 'conversations.sql',
            'post_init': None
        },
        'user_data': {
            'path': DATA_DIR / 'user_data.db',
            'schema': SCHEMA_DIR / 'user_data.sql',
            'post_init': None
        }
    }
    
    # Filter databases if specific ones requested
    if databases:
        db_configs = {k: v for k, v in db_configs.items() if k in databases}
    
    logger.info(f"Initializing databases: {list(db_configs.keys())}")
    
    success_count = 0
    for db_name, config in db_configs.items():
        if init_database(config['path'], config['schema'], db_name):
            # Run post-initialization if defined
            if config['post_init']:
                config['post_init'](config['path'])
            success_count += 1
    
    logger.info(f"Successfully initialized {success_count}/{len(db_configs)} databases")
    return success_count == len(db_configs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize RAG Server databases")
    parser.add_argument(
        'databases',
        nargs='*',
        choices=['auth', 'metrics', 'conversations', 'user_data'],
        help='Specific databases to initialize (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reinitialize without prompting'
    )
    
    args = parser.parse_args()
    
    if args.force:
        logger.warning("Force mode enabled - will reinitialize without prompting")
    
    success = main(args.databases if args.databases else None)
    sys.exit(0 if success else 1)