"""
Database-backed configuration management
Stores configuration settings in database for persistence and sharing
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigDB:
    """Database-backed configuration storage"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize config database"""
        if db_path is None:
            # Default to local SQLite database in webui directory
            db_path = Path(__file__).parent / "config.db"
        
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize the configuration database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        value_type TEXT NOT NULL DEFAULT 'json',
                        description TEXT,
                        category TEXT DEFAULT 'general',
                        is_sensitive BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        changed_by TEXT,
                        change_reason TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_config_key ON config_settings(key)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_config_category ON config_settings(category)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_key ON config_history(key)
                """)
                
                conn.commit()
                logger.info(f"Configuration database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize config database: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT value, value_type FROM config_settings WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result is None:
                    return default
                
                value, value_type = result
                
                # Deserialize based on type
                if value_type == 'json':
                    return json.loads(value)
                elif value_type == 'string':
                    return value
                elif value_type == 'int':
                    return int(value)
                elif value_type == 'float':
                    return float(value)
                elif value_type == 'bool':
                    return value.lower() in ('true', '1', 'yes')
                else:
                    return json.loads(value)  # Default to JSON
                    
        except Exception as e:
            logger.error(f"Failed to get config value for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any, description: str = "", category: str = "general", 
            is_sensitive: bool = False, updated_by: str = "", reason: str = ""):
        """Set a configuration value"""
        try:
            # Determine value type and serialize
            if isinstance(value, bool):
                value_type = 'bool'
                serialized_value = str(value).lower()
            elif isinstance(value, int):
                value_type = 'int'
                serialized_value = str(value)
            elif isinstance(value, float):
                value_type = 'float'
                serialized_value = str(value)
            elif isinstance(value, str):
                value_type = 'string'
                serialized_value = value
            else:
                value_type = 'json'
                serialized_value = json.dumps(value, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get old value for history
                old_value = self.get(key)
                
                # Insert or update setting
                conn.execute("""
                    INSERT OR REPLACE INTO config_settings 
                    (key, value, value_type, description, category, is_sensitive, updated_at, updated_by)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (key, serialized_value, value_type, description, category, is_sensitive, updated_by))
                
                # Record history if value changed
                if old_value != value:
                    conn.execute("""
                        INSERT INTO config_history 
                        (key, old_value, new_value, changed_by, change_reason, timestamp)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (key, json.dumps(old_value), json.dumps(value), updated_by, reason))
                
                conn.commit()
                logger.info(f"Config value set: {key} = {value} (type: {value_type})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set config value for key '{key}': {e}")
            return False
    
    def delete(self, key: str, deleted_by: str = "", reason: str = ""):
        """Delete a configuration value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get old value for history
                old_value = self.get(key)
                
                if old_value is not None:
                    # Record deletion in history
                    conn.execute("""
                        INSERT INTO config_history 
                        (key, old_value, new_value, changed_by, change_reason, timestamp)
                        VALUES (?, ?, NULL, ?, ?, CURRENT_TIMESTAMP)
                    """, (key, json.dumps(old_value), deleted_by, reason))
                
                # Delete setting
                cursor = conn.execute("DELETE FROM config_settings WHERE key = ?", (key,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Config value deleted: {key}")
                    return True
                else:
                    logger.warning(f"Config key not found for deletion: {key}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete config value for key '{key}': {e}")
            return False
    
    def list_all(self, category: Optional[str] = None, include_sensitive: bool = False) -> Dict[str, Any]:
        """List all configuration values"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if category:
                    if include_sensitive:
                        cursor = conn.execute(
                            "SELECT key, value, value_type FROM config_settings WHERE category = ?",
                            (category,)
                        )
                    else:
                        cursor = conn.execute(
                            "SELECT key, value, value_type FROM config_settings WHERE category = ? AND is_sensitive = FALSE",
                            (category,)
                        )
                else:
                    if include_sensitive:
                        cursor = conn.execute("SELECT key, value, value_type FROM config_settings")
                    else:
                        cursor = conn.execute("SELECT key, value, value_type FROM config_settings WHERE is_sensitive = FALSE")
                
                result = {}
                for row in cursor.fetchall():
                    key, value, value_type = row
                    
                    # Deserialize value
                    if value_type == 'json':
                        result[key] = json.loads(value)
                    elif value_type == 'string':
                        result[key] = value
                    elif value_type == 'int':
                        result[key] = int(value)
                    elif value_type == 'float':
                        result[key] = float(value)
                    elif value_type == 'bool':
                        result[key] = value.lower() in ('true', '1', 'yes')
                    else:
                        result[key] = json.loads(value)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to list config values: {e}")
            return {}
    
    def get_categories(self) -> List[str]:
        """Get all configuration categories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT category FROM config_settings ORDER BY category")
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def get_history(self, key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if key:
                    cursor = conn.execute("""
                        SELECT key, old_value, new_value, changed_by, change_reason, timestamp
                        FROM config_history 
                        WHERE key = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (key, limit))
                else:
                    cursor = conn.execute("""
                        SELECT key, old_value, new_value, changed_by, change_reason, timestamp
                        FROM config_history 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                history = []
                for row in cursor.fetchall():
                    key, old_value, new_value, changed_by, change_reason, timestamp = row
                    
                    # Parse JSON values
                    try:
                        old_val = json.loads(old_value) if old_value else None
                    except:
                        old_val = old_value
                    
                    try:
                        new_val = json.loads(new_value) if new_value else None
                    except:
                        new_val = new_value
                    
                    history.append({
                        'key': key,
                        'old_value': old_val,
                        'new_value': new_val,
                        'changed_by': changed_by,
                        'change_reason': change_reason,
                        'timestamp': timestamp
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get config history: {e}")
            return []
    
    def export_config(self, category: Optional[str] = None, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if category:
                    if include_sensitive:
                        cursor = conn.execute("""
                            SELECT key, value, value_type, description, category, is_sensitive
                            FROM config_settings 
                            WHERE category = ?
                        """, (category,))
                    else:
                        cursor = conn.execute("""
                            SELECT key, value, value_type, description, category, is_sensitive
                            FROM config_settings 
                            WHERE category = ? AND is_sensitive = FALSE
                        """, (category,))
                else:
                    if include_sensitive:
                        cursor = conn.execute("""
                            SELECT key, value, value_type, description, category, is_sensitive
                            FROM config_settings
                        """)
                    else:
                        cursor = conn.execute("""
                            SELECT key, value, value_type, description, category, is_sensitive
                            FROM config_settings 
                            WHERE is_sensitive = FALSE
                        """)
                
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'category': category,
                    'include_sensitive': include_sensitive,
                    'settings': {}
                }
                
                for row in cursor.fetchall():
                    key, value, value_type, description, cat, is_sensitive = row
                    
                    # Deserialize value
                    if value_type == 'json':
                        actual_value = json.loads(value)
                    elif value_type == 'string':
                        actual_value = value
                    elif value_type == 'int':
                        actual_value = int(value)
                    elif value_type == 'float':
                        actual_value = float(value)
                    elif value_type == 'bool':
                        actual_value = value.lower() in ('true', '1', 'yes')
                    else:
                        actual_value = json.loads(value)
                    
                    export_data['settings'][key] = {
                        'value': actual_value,
                        'type': value_type,
                        'description': description,
                        'category': cat,
                        'is_sensitive': bool(is_sensitive)
                    }
                
                return export_data
                
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return {}
    
    def import_config(self, config_data: Dict[str, Any], imported_by: str = "", 
                     overwrite: bool = False) -> bool:
        """Import configuration from dictionary"""
        try:
            if 'settings' not in config_data:
                logger.error("Invalid config data format - missing 'settings' key")
                return False
            
            success_count = 0
            total_count = len(config_data['settings'])
            
            for key, setting_info in config_data['settings'].items():
                if isinstance(setting_info, dict) and 'value' in setting_info:
                    # New format with metadata
                    value = setting_info['value']
                    description = setting_info.get('description', '')
                    category = setting_info.get('category', 'imported')
                    is_sensitive = setting_info.get('is_sensitive', False)
                else:
                    # Simple format - just value
                    value = setting_info
                    description = 'Imported setting'
                    category = 'imported'
                    is_sensitive = False
                
                # Check if key exists and overwrite policy
                if not overwrite and self.get(key) is not None:
                    logger.warning(f"Skipping existing key: {key}")
                    continue
                
                if self.set(key, value, description, category, is_sensitive, 
                          imported_by, "Configuration import"):
                    success_count += 1
            
            logger.info(f"Config import completed: {success_count}/{total_count} settings imported")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the configuration database"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Config database backed up to: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup config database: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore configuration database from backup"""
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Config database restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore config database: {e}")
            return False

# Global instance
config_db = ConfigDB()