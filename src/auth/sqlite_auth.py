"""
SQLite 기반 간단한 인증 시스템
메인 DB와 별도로 권한 관리만 SQLite로 처리
"""

import sqlite3
import hashlib
import jwt
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

class SQLiteAuthManager:
    """SQLite 기반 간단한 인증 관리자"""
    
    def __init__(self, db_path: str = "auth.db"):
        self.db_path = Path(db_path)
        self.secret_key = "your-secret-key-here"  # 실제로는 환경변수로 관리
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
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
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
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """JWT 토큰 생성"""
        payload = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # 세션 저장
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (user_id, token, expires_at)
                VALUES (?, ?, ?)
            """, (
                user_data['id'],
                token,
                datetime.now(timezone.utc) + timedelta(hours=24)
            ))
            conn.commit()
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # 세션 확인
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT s.*, u.username, u.email, u.role
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP
                """, (token,))
                
                session = cursor.fetchone()
                if session:
                    return dict(session)
            
        except jwt.ExpiredSignatureError:
            pass
        except jwt.InvalidTokenError:
            pass
        
        return None
    
    def logout(self, token: str) -> bool:
        """로그아웃 (토큰 무효화)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return cursor.rowcount > 0
    
    
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