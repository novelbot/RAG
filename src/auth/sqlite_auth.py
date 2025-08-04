"""
SQLite 기반 간단한 인증 시스템
메인 DB와 별도로 권한 관리만 SQLite로 처리
"""

import sqlite3
import hashlib
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

class SQLiteAuthManager:
    """SQLite 기반 간단한 인증 관리자"""
    
    def __init__(self, db_path: str = "auth.db"):
        self.db_path = Path(db_path)
        self._init_database()
        self._create_default_users()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            
            conn.commit()
    
    def _create_default_users(self):
        """기본 사용자 계정 생성"""
        default_users = [
            {"username": "admin", "password": "admin123", "role": "admin", "email": "admin@example.com"},
            {"username": "user", "password": "user123", "role": "user", "email": "user@example.com"},
            {"username": "manager", "password": "manager123", "role": "manager", "email": "manager@example.com"}
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for user_data in default_users:
                # 이미 존재하는지 확인
                cursor = conn.execute(
                    "SELECT id FROM users WHERE username = ?", 
                    (user_data["username"],)
                )
                if cursor.fetchone() is None:
                    password_hash = self._hash_password(user_data["password"])
                    conn.execute("""
                        INSERT INTO users (username, password_hash, email, role)
                        VALUES (?, ?, ?, ?)
                    """, (
                        user_data["username"],
                        password_hash,
                        user_data["email"],
                        user_data["role"]
                    ))
            conn.commit()
    
    def _hash_password(self, password: str) -> str:
        """패스워드 해싱"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str = "", role: str = "user") -> Optional[Dict[str, Any]]:
        """새 사용자 생성"""
        password_hash = self._hash_password(password)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # 사용자명 중복 확인
                cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    return None  # 이미 존재하는 사용자
                
                # 새 사용자 생성
                conn.execute("""
                    INSERT INTO users (username, password_hash, email, role)
                    VALUES (?, ?, ?, ?)
                """, (username, password_hash, email, role))
                
                # 생성된 사용자 정보 조회
                cursor = conn.execute("""
                    SELECT id, username, email, role, is_active
                    FROM users WHERE username = ?
                """, (username,))
                
                new_user = cursor.fetchone()
                if new_user:
                    return dict(new_user)
                
        except sqlite3.IntegrityError:
            return None  # 중복 데이터
        
        return None
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """사용자 인증"""
        password_hash = self._hash_password(password)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, username, email, role, is_active
                FROM users 
                WHERE username = ? AND password_hash = ? AND is_active = 1
            """, (username, password_hash))
            
            user = cursor.fetchone()
            if user:
                # 마지막 로그인 시간 업데이트
                conn.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (user['id'],))
                conn.commit()
                
                return dict(user)
        
        return None
    
    
    
    
    
    def get_users(self) -> List[Dict[str, Any]]:
        """모든 사용자 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, username, email, role, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    

# 전역 인스턴스
auth_manager = SQLiteAuthManager()