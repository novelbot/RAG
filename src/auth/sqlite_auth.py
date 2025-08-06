"""
SQLite 기반 간단한 인증 시스템
메인 DB와 별도로 권한 관리만 SQLite로 처리
"""

import sqlite3
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import bcrypt
from dotenv import load_dotenv
import hashlib  # SHA256 호환성을 위해 유지

# Load environment variables
load_dotenv()

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
                    last_login TIMESTAMP,
                    password_updated_at TIMESTAMP,
                    force_password_change BOOLEAN DEFAULT 0
                )
            """)
            
            
            conn.commit()
    
    def _create_default_users(self):
        """기본 사용자 계정 생성"""
        # 환경변수에서 초기 비밀번호 가져오기 (없으면 안전한 기본값 사용)
        admin_password = os.getenv('INITIAL_ADMIN_PASSWORD', 'ChangeMe!Admin2024')
        user_password = os.getenv('INITIAL_USER_PASSWORD', 'ChangeMe!User2024')
        manager_password = os.getenv('INITIAL_MANAGER_PASSWORD', 'ChangeMe!Manager2024')
        
        default_users = [
            {"username": "admin", "password": admin_password, "role": "admin", "email": "admin@example.com"},
            {"username": "user", "password": user_password, "role": "user", "email": "user@example.com"},
            {"username": "manager", "password": manager_password, "role": "manager", "email": "manager@example.com"}
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
        """패스워드 해싱 (bcrypt 사용)"""
        # bcrypt는 자동으로 salt를 생성하고 포함시킵니다
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """패스워드 검증 (bcrypt 및 SHA256 호환)"""
        try:
            # SHA256 해시 형식인지 확인 (64자 hex string)
            if len(password_hash) == 64 and all(c in '0123456789abcdef' for c in password_hash.lower()):
                # 기존 SHA256 해시와 비교 (마이그레이션 전)
                return hashlib.sha256(password.encode()).hexdigest() == password_hash
            else:
                # bcrypt 해시와 비교
                return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
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
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # 먼저 사용자 정보 가져오기
            cursor = conn.execute("""
                SELECT id, username, email, role, is_active, password_hash, force_password_change
                FROM users 
                WHERE username = ? AND is_active = 1
            """, (username,))
            
            user = cursor.fetchone()
            if user:
                # 비밀번호 검증
                if self._verify_password(password, user['password_hash']):
                    # 마지막 로그인 시간 업데이트
                    conn.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (user['id'],))
                    
                    # SHA256 해시를 bcrypt로 마이그레이션
                    if len(user['password_hash']) == 64:  # SHA256 해시인 경우
                        new_hash = self._hash_password(password)
                        conn.execute("""
                            UPDATE users 
                            SET password_hash = ?, password_updated_at = CURRENT_TIMESTAMP 
                            WHERE id = ?
                        """, (new_hash, user['id']))
                    
                    conn.commit()
                    
                    # 비밀번호 해시 제외한 사용자 정보 반환
                    user_dict = dict(user)
                    user_dict.pop('password_hash', None)
                    return user_dict
        
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