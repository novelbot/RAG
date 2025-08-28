# NovelBot RAG Server 🚀

웹 소설 콘텐츠를 위한 고성능 RAG(Retrieval-Augmented Generation) 서버 - LangChain 기반 아키텍처와 실시간 스트리밍을 지원하는 차세대 AI 시스템

## 목차

- [핵심 기능](#핵심-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [빠른 시작](#빠른-시작)
- [API 문서](#api-문서)
- [설정 가이드](#설정-가이드)
- [고급 기능](#고급-기능)
- [CLI 명령어](#cli-명령어)
- [개발 가이드](#개발-가이드)
- [문제 해결](#문제-해결)

## 핵심 기능

### 고성능 벡터 검색
- **Milvus 벡터 데이터베이스**: 대규모 벡터 데이터의 실시간 검색
- **다중 인덱스 지원**: IVF_FLAT, HNSW 등 최적화된 인덱싱
- **메타데이터 필터링**: 에피소드, 소설별 정밀 검색

### 다중 LLM 프로바이더
- **Google Gemini 2.0 Flash**: 최신 고속 모델 지원
- **OpenAI GPT-4**: GPT-4o-mini, GPT-4 모델 등 openAI 계열 모델 지원
- **Ollama**: 로컬 LLM 실행 (Llama, Mistral 등)
- **자동 폴백**: 프로바이더 장애 시 자동 전환

### 실시간 스트리밍
- **Server-Sent Events (SSE)**: 실시간 응답 스트리밍
- **타이핑 효과**: 자연스러운 대화 경험
- **프로그레스 트래킹**: 처리 상태 실시간 업데이트
- **에러 스트리밍**: 오류도 실시간으로 전달

### 대화 컨텍스트 관리
- **세션 기반 대화**: conversation_id로 대화 연속성 유지
- **컨텍스트 윈도우**: 자동 컨텍스트 크기 관리
- **대화 영구 저장**: SQLite 기반 대화 기록 저장
- **멀티턴 지원**: 여러 차례 대화 지원

### 보안 및 인증
- **JWT 토큰**: 60분 유효기간의 액세스 토큰
- **RBAC**: 역할 기반 접근 제어
- **비밀번호 해싱**: bcrypt 기반 안전한 비밀번호 저장
- **HTTPS/SSL**: TLS 암호화 지원

### 모니터링 및 디버깅
- **실시간 메트릭**: 성능 및 사용량 추적
- **헬스체크**: 시스템 상태 모니터링
- **프롬프트 디버깅**: LLM 프롬프트 추적 및 분석
- **토큰 사용량**: 비용 최적화를 위한 토큰 추적

## 시스템 아키텍처

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│    backend     │────▶│   FastAPI      │────▶│    Milvus      │
│  (Streaming)   │ SSE │   Server       │     │  Vector DB     │
└────────────────┘     └────────────────┘     └────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────▼──┐ ┌───▼───┐ ┌──▼──────────┐
              │ SQLite │ │  LLM  │ │  LangChain  │
              │  DBs   │ │  APIs │ │     RAG     │
              └────────┘ └───────┘ └─────────────┘
                   │          │            │
         ┌─────────┼──────────┼────────────┤
         │         │          │            │
    ┌────▼───┐ ┌──▼───┐ ┌───▼───┐ ┌──────▼──────┐
    │ Auth   │ │Metrics│ │ Conv. │ │ User Data   │
    │  DB    │ │  DB   │ │  DB   │ │     DB      │
    └────────┘ └───────┘ └───────┘ └─────────────┘
```

## 빠른 시작

### 사전 요구사항

- Python 3.11 이상
- UV 패키지 매니저
- Docker & Docker Compose
- 8GB 이상 RAM 권장

### 1. 프로젝트 클론

```bash
git clone https://github.com/novelbot/RAG.git
cd RAG
```

### 2. UV 설치 및 의존성 설정

```bash
# UV 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

### 3. 환경 변수 설정

`.env` 파일 생성:

```env
# LLM 설정 예시
LLM_PROVIDER=Google
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your-google-api-key

# 임베딩 설정 예시
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-api-key

# Milvus 설정 예시
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=yourpassword

# 서버 설정
API_HOST=0.0.0.0
API_PORT=8000

# JWT 비밀키
SECRET_KEY=your-secret-key-here

### 4. Milvus 벡터 데이터베이스 시작

```bash
docker-compose up -d milvus
```

### 5. 데이터베이스 초기화

```bash
# 모든 SQLite 데이터베이스 초기화
uv run rag-cli database init --sqlite

# 기본 관리자 계정 생성됨:
# Username: admin
# Password: admin123
```

### 6. 서버 시작

```bash
uv run rag-cli serve
# 서버 주소: http://localhost:8000
```

### 7. API 테스트

```bash
# 로그인
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 토큰을 받아서 이후 요청에 사용
export TOKEN="받은_토큰_값"

# 채팅 요청
curl -X POST http://localhost:8000/api/v1/episode/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요, 테스트 메시지입니다.",
    "episode_ids": [],
    "novel_ids": []
  }'
```

## API 문서

### 인증 API (`/api/v1/auth`)

#### 로그인
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}

Response:
{
  "access_token": "eyJhbGc...",
  "refresh_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### 사용자 등록
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepassword123"
}
```

#### 토큰 갱신
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGc..."
}
```

#### 현재 사용자 정보
```http
GET /api/v1/auth/me
Authorization: Bearer {token}

Response:
{
  "user_id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "roles": ["admin"],
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 에피소드 API (`/api/v1/episode`)

#### 일반 채팅 (JSON 응답)
```http
POST /api/v1/episode/chat
Authorization: Bearer {token}
Content-Type: application/json

{
  "message": "주인공의 첫 등장 장면을 설명해주세요",
  "episode_ids": [1, 2, 3],
  "novel_ids": [],
  "conversation_id": "optional-uuid",
  "use_conversation_context": true
}

Response:
{
  "response": "주인공은 첫 에피소드에서...",
  "conversation_id": "uuid",
  "sources": [...],
  "tokens_used": 1234
}
```

#### 스트리밍 채팅 (SSE)
```http
POST /api/v1/episode/chat/stream
Authorization: Bearer {token}
Content-Type: application/json

{
  "message": "이야기를 계속 들려주세요",
  "conversation_id": "existing-uuid",
  "use_conversation_context": true,
  "episode_ids": [],
  "novel_ids": []
}

Response (Server-Sent Events):
data: {"type": "start", "conversation_id": "uuid"}
data: {"type": "token", "content": "주인공은"}
data: {"type": "token", "content": " 첫"}
data: {"type": "token", "content": " 에피소드에서"}
data: {"type": "end", "tokens_used": 1234}
```

#### 벡터 검색
```http
POST /api/v1/episode/search
Authorization: Bearer {token}
Content-Type: application/json

{
  "query": "검색할 내용",
  "episode_ids": [],
  "novel_ids": [],
  "limit": 5,
  "similarity_threshold": 0.7
}

Response:
{
  "results": [
    {
      "content": "매칭된 텍스트...",
      "episode_id": 1,
      "similarity_score": 0.89,
      "metadata": {...}
    }
  ],
  "total": 5
}
```

#### 대화 조회
```http
GET /api/v1/episode/conversation/{conversation_id}
Authorization: Bearer {token}

Response:
{
  "conversation_id": "uuid",
  "messages": [
    {
      "role": "user",
      "content": "안녕하세요",
      "timestamp": "2024-01-01T00:00:00Z"
    },
    {
      "role": "assistant",
      "content": "안녕하세요! 무엇을 도와드릴까요?",
      "timestamp": "2024-01-01T00:00:01Z"
    }
  ],
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### 프롬프트 디버깅
```http
GET /api/v1/episode/debug/prompt/{conversation_id}
Authorization: Bearer {token}

Response:
{
  "conversation_id": "uuid",
  "last_prompt": {
    "system": "You are a helpful assistant...",
    "messages": [...],
    "context": "Retrieved documents...",
    "total_tokens": 2345
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### 에피소드 일괄 처리
```http
POST /api/v1/episode/process-all
Authorization: Bearer {token}
Content-Type: application/json

{
  "filter_episode_ids": [1, 2, 3],
  "chunk_size": 500,
  "overlap": 50
}

Response:
{
  "processed": 3,
  "success": 3,
  "failed": 0,
  "details": [...]
}
```

### 모니터링 API

#### 헬스체크
```http
GET /health

Response:
{
  "status": "healthy"
}
```

#### 상세 시스템 상태
```http
GET /api/v1/monitoring/health

Response:
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "milvus": "connected",
    "llm": "available"
  },
  "metrics": {
    "cpu_usage": 23.5,
    "memory_usage": 45.2,
    "active_connections": 10
  }
}
```

## 설정 가이드

### LLM 프로바이더 설정

#### Google Gemini (권장)
```env
LLM_PROVIDER=Google
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your-google-api-key
```

#### OpenAI GPT
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key
```

#### Ollama (로컬)
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

### 임베딩 프로바이더 설정

#### OpenAI Embeddings
```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-openai-api-key
```

#### Google Embeddings
```env
EMBEDDING_PROVIDER=Google
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=your-google-api-key
```

#### Ollama Embeddings (로컬)
```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=bge-m3
# 다른 옵션: nomic-embed-text, mxbai-embed-large
```

### RAG 설정

```env
# 검색 설정
RAG_RETRIEVAL_K=5              # 검색할 문서 수
RAG_SIMILARITY_THRESHOLD=0.7   # 최소 유사도 점수

# 벡터 차원 (자동 감지되지만 수동 설정 가능)
# VECTOR_DIMENSION=1536  # OpenAI
# VECTOR_DIMENSION=768   # Google
```

### 데이터베이스 설정

```env
# SQLite 경로 (기본값)
AUTH_DB_PATH=auth.db
METRICS_DB_PATH=metrics.db
CONVERSATIONS_DB_PATH=data/conversations.db
USER_DATA_DB_PATH=data/user_data.db

# MySQL (예시)
DB_DRIVER=mysql+pymysql
DB_HOST=localhost
DB_PORT=3306
DB_NAME=novelbot
DB_USER=dbuser
DB_PASSWORD=dbpassword
```

## CLI 명령어

### 서버 관리

```bash
# 서버 시작 (기본)
uv run rag-cli serve

# 특정 포트로 시작
uv run rag-cli serve --port 8080

# 디버그 모드
uv run rag-cli serve --debug
```

### 데이터베이스 관리

```bash
# 데이터베이스 초기화
uv run rag-cli database init --sqlite

# 특정 데이터베이스만 초기화
uv run rag-cli database init --sqlite auth metrics

# 데이터베이스 상태 확인
uv run rag-cli database status

# 강제 재초기화 (주의: 모든 데이터 삭제)
uv run rag-cli database init --sqlite --force
```


## 개발 가이드

### 프로젝트 구조

```
novelbot_RAG_server/
├── src/
│   ├── api/              # API 엔드포인트
│   │   ├── routes/       # 라우트 정의
│   │   │   ├── auth.py   # 인증 API
│   │   │   ├── episode.py # 에피소드 API
│   │   │   └── monitoring.py # 모니터링 API
│   │   └── middleware.py # 미들웨어
│   │
│   ├── auth/             # 인증 시스템
│   │   ├── jwt_manager.py # JWT 토큰 관리
│   │   ├── rbac.py      # 역할 기반 접근 제어
│   │   └── models.py    # 인증 모델
│   │
│   ├── llm/              # LLM 프로바이더
│   │   ├── langchain_providers.py # LangChain 통합
│   │   └── providers/    # 개별 프로바이더
│   │
│   ├── embedding/        # 임베딩 프로바이더
│   │   ├── factory.py    # 임베딩 팩토리
│   │   └── providers/    # 개별 프로바이더
│   │
│   ├── rag/              # RAG 파이프라인
│   │   ├── langchain_rag.py # LangChain RAG
│   │   ├── vector_search_engine.py # 벡터 검색
│   │   └── context_retriever.py # 컨텍스트 검색
│   │
│   ├── milvus/           # Milvus 통합
│   │   ├── client.py     # Milvus 클라이언트
│   │   ├── collection.py # 컬렉션 관리
│   │   └── search.py     # 검색 로직
│   │
│   ├── conversation/     # 대화 관리
│   │   └── storage.py    # 대화 저장소
│   │
│   └── core/             # 핵심 기능
│       ├── config.py     # 설정 관리
│       ├── logging.py    # 로깅 설정
│       └── database.py   # 데이터베이스 연결
│
├── database/             # 데이터베이스 스키마
│   └── schemas/         # SQL 스키마 파일
│
├── scripts/             # 유틸리티 스크립트
│   ├── generate_ssl_cert.sh # SSL 인증서 생성
│   └── init_databases.py # DB 초기화
│
├── templates/           # HTML 템플릿
│   └── test_streaming.html # 스트리밍 테스트 UI
│
├── docker-compose.yml   # Docker 구성
├── pyproject.toml      # 프로젝트 설정
└── .env                # 환경 변수
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**NovelBot RAG Server**