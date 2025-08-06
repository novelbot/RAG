#!/usr/bin/env python3
"""
Auth database migration script.
Adds new columns and migrates existing passwords from SHA256 to bcrypt.
"""

import sqlite3
import bcrypt
import hashlib
from pathlib import Path

def migrate_auth_database():
    """기존 auth.db에 새 컬럼 추가 및 비밀번호 마이그레이션"""
    
    db_path = Path("auth.db")
    if not db_path.exists():
        print("auth.db not found. Creating new database...")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("Starting database migration...")
    
    # 1. 새 컬럼 추가 (이미 존재하면 무시)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN password_updated_at TIMESTAMP")
        print("✓ Added password_updated_at column")
    except sqlite3.OperationalError:
        print("- password_updated_at column already exists")
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN force_password_change BOOLEAN DEFAULT 0")
        print("✓ Added force_password_change column")
    except sqlite3.OperationalError:
        print("- force_password_change column already exists")
    
    # 2. SHA256 해시를 사용하는 기존 사용자들에게 비밀번호 변경 플래그 설정
    cursor.execute("""
        SELECT id, username, password_hash 
        FROM users
    """)
    
    users = cursor.fetchall()
    sha256_users = []
    
    for user in users:
        # SHA256 해시는 64자의 hex string
        if len(user['password_hash']) == 64:
            sha256_users.append(user['username'])
            # 기존 SHA256 사용자는 다음 로그인 시 비밀번호 변경 권장
            cursor.execute("""
                UPDATE users 
                SET force_password_change = 1 
                WHERE id = ?
            """, (user['id'],))
    
    if sha256_users:
        print(f"✓ Marked {len(sha256_users)} users for password update: {', '.join(sha256_users)}")
        print("  These users will be prompted to change password on next login")
    else:
        print("✓ All users already using bcrypt")
    
    conn.commit()
    conn.close()
    
    print("\nMigration completed successfully!")
    print("\nNote: Users with SHA256 passwords will automatically migrate to bcrypt")
    print("      when they next log in with their existing password.")

if __name__ == "__main__":
    migrate_auth_database()