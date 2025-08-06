# NovelBot RAG Server - 프로젝트 구조

## 📌 프로젝트 개요

**NovelBot RAG Server**는 웹소설 에피소드 전용 RAG(Retrieval-Augmented Generation) 시스템입니다.
Milvus 벡터 데이터베이스를 기반으로 웹소설 에피소드를 효율적으로 검색하고 처리합니다.

### 핵심 특징
- 🎯 **웹소설 에피소드 특화**: 에피소드 단위 검색 및 처리 최적화
- 🤖 **다중 LLM 지원**: OpenAI, Anthropic, Google Gemini, Ollama
- 📊 **다중 임베딩 지원**: Ollama, OpenAI, Google 임베딩 모델
- 🔐 **JWT 기반 인증**: 안전한 API 접근 제어
- ⚡ **고성능 벡터 검색**: Milvus 기반 유사도 검색

## 🏗️ 디렉토리 구조

```
novelbot_RAG_server/
│
├── 📄 main.py                    # 애플리케이션 진입점
├── 🔧 .env                       # 환경 변수 설정
├── 📦 pyproject.toml             # 프로젝트 의존성 정의
├── 🔒 uv.lock                    # 의존성 버전 고정
│
├── 💾 data/                      # 데이터 저장소
│   ├── auth.db                   # 사용자 인증 데이터베이스
│   └── metrics.db                # 시스템 메트릭 데이터베이스
│
└── 📂 src/                       # 소스 코드
    ├── 🌐 api/                   # API 엔드포인트
    │   ├── routes/               
    │   │   ├── auth.py           # 인증 엔드포인트
    │   │   ├── episode.py        # 에피소드 RAG 엔드포인트
    │   │   └── monitoring.py     # 헬스체크 및 상태
    │   └── schemas.py            # Pydantic 스키마
    │
    ├── 🔑 auth/                  # 인증 시스템
    │   ├── dependencies.py       # 인증 의존성
    │   ├── jwt_manager.py        # JWT 토큰 관리
    │   └── sqlite_auth.py        # SQLite 기반 인증
    │
    ├── 💻 cli/                   # CLI 인터페이스
    │   ├── main.py               # CLI 진입점
    │   └── commands/             
    │       ├── data.py           # 데이터 수집 명령
    │       ├── serve.py          # 서버 실행 명령
    │       └── user.py           # 사용자 관리 명령
    │
    ├── ⚙️ core/                   # 핵심 설정
    │   ├── app.py                # FastAPI 앱 팩토리
    │   ├── config.py             # 설정 관리
    │   ├── database.py           # DB 연결 관리
    │   └── logging.py            # 로깅 설정
    │
    ├── 🗄️ database/               # 데이터베이스 어댑터
    │   ├── base.py               # 베이스 DB 매니저
    │   └── mysql.py              # MySQL 어댑터
    │
    ├── 🧮 embedding/              # 임베딩 시스템
    │   ├── base.py               # 베이스 임베딩 클래스
    │   ├── manager.py            # 임베딩 매니저
    │   └── providers/
    │       ├── ollama.py         # Ollama 임베딩
    │       ├── openai.py         # OpenAI 임베딩
    │       └── google.py         # Google 임베딩
    │
    ├── 📚 episode/                # 에피소드 처리 (핵심)
    │   ├── manager.py            # 에피소드 매니저
    │   ├── processor.py          # 에피소드 처리기
    │   ├── search_engine.py      # 에피소드 검색 엔진
    │   └── vector_store.py       # 벡터 저장소 인터페이스
    │
    ├── 🤖 llm/                    # LLM 시스템
    │   ├── base.py               # 베이스 LLM 클래스
    │   ├── manager.py            # LLM 매니저
    │   └── providers/
    │       ├── ollama.py         # Ollama LLM
    │       ├── openai.py         # OpenAI GPT
    │       ├── claude.py         # Anthropic Claude
    │       └── gemini.py         # Google Gemini
    │
    ├── 📊 metrics/                # 메트릭 수집
    │   ├── collectors.py         # 메트릭 수집기
    │   └── database.py           # 메트릭 DB
    │
    ├── 🔍 milvus/                 # Milvus 벡터 DB
    │   ├── client.py             # Milvus 클라이언트
    │   ├── collection.py         # 컬렉션 관리
    │   ├── schema.py             # 스키마 정의
    │   └── search.py             # 벡터 검색
    │
    ├── 📋 models/                 # 데이터 모델
    │   ├── base.py               # 베이스 모델
    │   ├── episode.py            # 에피소드 모델
    │   └── user.py               # 사용자 모델
    │
    ├── 🎯 rag/                    # RAG 코어
    │   ├── context_builder.py    # 컨텍스트 구성
    │   ├── query_preprocessor.py # 쿼리 전처리
    │   └── vector_search_engine.py # 벡터 검색 엔진
    │
    ├── 🛠️ services/               # 비즈니스 서비스
    │   ├── data_sync.py          # 데이터 동기화
    │   └── query_logger.py       # 쿼리 로깅
    │
    ├── ✂️ text_processing/        # 텍스트 처리
    │   ├── text_cleaner.py       # 텍스트 정제
    │   └── text_splitter.py      # 텍스트 분할
    │
    └── 🔧 utils/                  # 유틸리티
        ├── embeddings.py          # 임베딩 유틸
        └── validators.py          # 검증 유틸

```

## 🔑 주요 API 엔드포인트

### 인증 (`/api/v1/auth`)
- `POST /login` - 로그인
- `POST /register` - 회원가입
- `POST /logout` - 로그아웃
- `POST /refresh` - 토큰 갱신
- `GET /me` - 현재 사용자 정보

### 에피소드 RAG (`/api/v1/episode`)
- `POST /chat` - 에피소드 기반 대화
- `POST /search` - 에피소드 검색
- `POST /process` - 에피소드 처리
- `POST /process-all` - 전체 에피소드 처리

### 모니터링 (`/api/v1/monitoring`)
- `GET /health` - 종합 헬스체크
- `GET /health/simple` - 간단한 헬스체크
- `GET /status` - 서비스 상태

## ⚙️ 환경 설정

### 필수 환경 변수 (`.env`)
```bash
# 데이터베이스
DB_HOST=localhost
DB_PORT=3306
DB_NAME=novelbot
DB_USER=root
DB_PASSWORD=password

# Milvus 벡터 DB
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=episode_vectors

# 임베딩 설정
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

# LLM 설정
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2

# JWT 인증
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🚀 실행 방법

### 서버 실행
```bash
# 의존성 설치
uv sync

# 서버 실행
uv run main.py

# 또는 CLI를 통한 실행
uv run rag-cli serve --reload
```

### CLI 명령어
```bash
# 사용자 관리
uv run rag-cli user create --username admin --password admin123 --role admin

# 에피소드 데이터 수집
uv run rag-cli data ingest --episode-mode --database

# 설정 확인
uv run rag-cli config show
```

## 📊 데이터 흐름

### 에피소드 처리 파이프라인
```
MySQL DB (에피소드) 
    → 텍스트 청킹 
    → 임베딩 생성 (Ollama/OpenAI)
    → Milvus 저장
    → 벡터 인덱싱
```

### RAG 쿼리 흐름
```
사용자 쿼리 
    → 쿼리 임베딩 
    → Milvus 벡터 검색 
    → 컨텍스트 구성 
    → LLM 응답 생성
```

## 🔐 인증 시스템

- **JWT 기반 인증**: 상태 비저장 토큰 인증
- **SQLite 사용자 DB**: 경량 사용자 관리
- **역할 기반 접근**: user, admin 역할 지원

## 📈 시스템 요구사항

### 최소 요구사항
- Python 3.11+
- MySQL 5.7+ 또는 MariaDB 10.3+
- Milvus 2.3+
- 4GB RAM
- 10GB 디스크 공간

### 권장 사양
- Python 3.12
- MySQL 8.0+
- Milvus 2.4+
- 16GB RAM
- SSD 50GB+

## 🎯 프로젝트 특징

1. **웹소설 특화**: 에피소드 단위 처리 최적화
2. **모듈러 설계**: 교체 가능한 LLM/임베딩 프로바이더
3. **고성능**: 벡터 검색 기반 빠른 응답
4. **확장 가능**: 대량 에피소드 처리 지원
5. **보안**: JWT 인증 및 안전한 API

## 📝 개발 가이드

### 코드 스타일
- Python 3.11+ 타입 힌트 사용
- Black 코드 포매터
- 비동기 처리 우선

### 주요 의존성
- **FastAPI**: 웹 프레임워크
- **LangChain**: LLM 오케스트레이션  
- **PyMilvus**: 벡터 DB 클라이언트
- **SQLAlchemy**: ORM
- **Pydantic**: 데이터 검증

---

*이 문서는 NovelBot RAG Server의 현재 구조를 반영합니다. (2025년 1월 기준)*