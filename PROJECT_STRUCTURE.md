# NovelBot RAG Server 프로젝트 구조 분석

## 📌 프로젝트 개요

**NovelBot RAG Server**는 Milvus 벡터 데이터베이스를 기반으로 한 프로덕션 레디 RAG(Retrieval-Augmented Generation) 서버입니다. 웹소설 에피소드 데이터를 처리하고, 다중 LLM 지원과 세밀한 접근 제어(FGAC) 기능을 제공합니다.

### 주요 특징
- 🤖 **다중 LLM 지원**: OpenAI, Anthropic, Google Gemini, Ollama
- 📊 **다중 데이터베이스 지원**: PostgreSQL, MySQL, MariaDB, Oracle, SQL Server
- 🔐 **세밀한 접근 제어**: JWT 기반 인증 및 RBAC
- 🎯 **이중 RAG 운영 모드**: 단일/다중 LLM 모드
- 📚 **에피소드 기반 검색**: 웹소설 에피소드 특화 RAG 시스템

## 🏗️ 프로젝트 디렉토리 구조

```
novelbot_RAG_server/
│
├── 📦 src/                        # 핵심 소스 코드
│   ├── 🔐 access_control/         # 접근 제어 및 권한 관리
│   ├── 🌐 api/                    # FastAPI 엔드포인트
│   ├── 🔑 auth/                   # 인증 및 인가
│   ├── 💻 cli/                    # CLI 명령어 인터페이스
│   ├── ⚙️ core/                   # 핵심 설정 및 유틸리티
│   ├── 🗄️ database/               # 데이터베이스 연결 관리
│   ├── 🧮 embedding/              # 임베딩 프로바이더
│   ├── 📖 episode/                # 에피소드 처리 및 RAG
│   ├── 📄 extraction/             # 문서 추출
│   ├── 📁 file_system/            # 파일 시스템 처리
│   ├── 🤖 llm/                    # LLM 프로바이더
│   ├── 📊 metrics/                # 메트릭 수집
│   ├── 🔍 milvus/                 # Milvus 벡터 DB 클라이언트
│   ├── 📋 models/                 # 데이터 모델
│   ├── 🔄 pipeline/               # 데이터 처리 파이프라인
│   ├── 🎯 rag/                    # RAG 코어 로직
│   ├── 💬 response_generation/    # 응답 생성
│   ├── 🛠️ services/               # 비즈니스 서비스
│   ├── ✂️ text_processing/        # 텍스트 처리
│   └── 🔧 utils/                  # 유틸리티 함수
│
├── 🌐 webui/                      # Streamlit 웹 UI
│   ├── 📱 app.py                  # 메인 애플리케이션
│   ├── 📄 pages/                  # UI 페이지들
│   └── ⚙️ config.py               # UI 설정
│
├── 🧪 tests/                      # 테스트 코드
│   ├── unit/                      # 단위 테스트
│   └── integration/               # 통합 테스트
│
├── 📚 docs/                       # 문서
├── 📝 examples/                   # 사용 예제
└── 📊 logs/                       # 로그 파일

```

## 🔧 기술 스택

### 핵심 프레임워크
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **LangChain**: LLM 오케스트레이션
- **Pydantic**: 데이터 검증 및 스키마 관리
- **SQLAlchemy**: ORM 및 데이터베이스 관리

### 벡터 데이터베이스
- **Milvus**: 벡터 저장 및 유사도 검색
- **PyMilvus**: Milvus Python 클라이언트

### LLM 프로바이더
- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude 모델
- **Google**: Gemini 모델
- **Ollama**: 로컬 LLM 지원

### 임베딩 모델
- **Ollama**: `jeffh/intfloat-multilingual-e5-large-instruct:f32` (권장)
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`
- **Google**: Gemini 임베딩

### 인증 및 보안
- **JWT**: JSON Web Token 기반 인증
- **Passlib + bcrypt**: 비밀번호 해싱
- **python-jose**: JWT 토큰 처리

### UI 및 시각화
- **Streamlit**: 웹 UI 프레임워크
- **Plotly**: 데이터 시각화

## 📋 주요 모듈 설명

### 1. API 모듈 (`src/api/`)
```
api/
├── routes/
│   ├── episode.py      # 에피소드 기반 RAG 엔드포인트
│   ├── query.py        # 일반 쿼리 처리
│   ├── auth.py         # 인증 관련 엔드포인트
│   ├── documents.py    # 문서 관리
│   └── monitoring.py   # 모니터링 엔드포인트
├── schemas.py          # Pydantic 스키마 정의
└── middleware.py       # FastAPI 미들웨어
```

#### 주요 엔드포인트
- `POST /api/v1/episodes/chat`: 에피소드 기반 대화
- `POST /api/v1/episodes/search`: 에피소드 검색
- `POST /api/v1/query/search`: 일반 검색
- `POST /api/v1/auth/login`: 사용자 로그인
- `POST /api/v1/auth/refresh`: 토큰 갱신

### 2. Episode 모듈 (`src/episode/`)
웹소설 에피소드 처리를 위한 특화 모듈:

```python
episode/
├── manager.py          # 에피소드 매니저
├── processor.py        # 에피소드 임베딩 처리
├── search_engine.py    # 에피소드 검색 엔진
└── vector_store.py     # 벡터 저장소 인터페이스
```

### 3. 임베딩 시스템 (`src/embedding/`)
```python
embedding/
├── base.py             # 베이스 임베딩 클래스
├── manager.py          # 임베딩 매니저
├── factory.py          # 프로바이더 팩토리
└── providers/
    ├── ollama.py       # Ollama 임베딩 (로컬)
    ├── openai.py       # OpenAI 임베딩
    └── google.py       # Google 임베딩
```

#### Ollama 임베딩 모델 지원
- `nomic-embed-text`: 768차원
- `mxbai-embed-large`: 1024차원
- `jeffh/intfloat-multilingual-e5-large-instruct`: 1024차원 (한국어 최적화)
- `bge-m3`: 768차원 (다국어)

### 4. LLM 시스템 (`src/llm/`)
```python
llm/
├── base.py             # 베이스 LLM 클래스
├── manager.py          # LLM 매니저
└── providers/
    ├── ollama.py       # Ollama (로컬 LLM)
    ├── openai.py       # OpenAI GPT
    ├── claude.py       # Anthropic Claude
    └── gemini.py       # Google Gemini
```

### 5. Milvus 통합 (`src/milvus/`)
```python
milvus/
├── client.py           # Milvus 클라이언트
├── collection.py       # 컬렉션 관리
├── schema.py           # 스키마 정의
├── search.py           # 벡터 검색
└── rbac.py            # 역할 기반 접근 제어
```

#### Milvus 스키마 구조
```python
{
    "id": INT64 (Primary Key),
    "content": VARCHAR,
    "embedding": FLOAT_VECTOR,
    "metadata": JSON,
    "owner_id": VARCHAR,
    "group_ids": ARRAY[VARCHAR],
    "permission_level": INT32
}
```

## 🔄 데이터 처리 파이프라인

### 1. 문서 수집 (Ingestion)
```
파일/DB → 추출(Extraction) → 청크 분할(Chunking) → 임베딩 생성
```

### 2. RAG 처리 흐름
```
사용자 쿼리 → 쿼리 임베딩 → 벡터 검색 → 컨텍스트 검색 → LLM 생성 → 응답
```

### 3. 에피소드 처리 흐름
```python
# 에피소드 데이터 구조
{
    "novel_id": int,
    "episode_id": int,
    "episode_number": int,
    "title": str,
    "content": str,
    "characters": List[str],
    "embedding": List[float]
}
```

## 🔐 인증 및 권한 시스템

### JWT 토큰 구조
```python
{
    "sub": "user_id",
    "username": "username",
    "roles": ["user", "admin"],
    "exp": 1234567890
}
```

### 권한 레벨
- **Guest**: 읽기 전용
- **User**: 기본 사용자 권한
- **Premium**: 프리미엄 기능 접근
- **Admin**: 전체 시스템 관리

## ⚙️ 환경 설정

### 필수 환경 변수 (`.env`)
```bash
# 데이터베이스
DB_HOST=localhost
DB_PORT=3306
DB_NAME=novelbot
DB_USER=root
DB_PASSWORD=password

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_vectors

# 임베딩 설정
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=jeffh/intfloat-multilingual-e5-large-instruct:f32
EMBEDDING_API_KEY=

# LLM 설정
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:27b-it-q8_0
LLM_API_KEY=

# JWT 설정
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## 🚀 실행 방법

### 1. 개발 서버 실행
```bash
# uv를 사용한 의존성 설치
uv sync

# 서버 실행
uv run main.py

# 또는 CLI를 통한 실행
uv run rag-cli serve --reload
```

### 2. Web UI 실행
```bash
streamlit run webui/app.py
```

### 3. CLI 명령어
```bash
# 사용자 관리
uv run rag-cli user create --username admin --password admin123 --role admin

# 데이터 수집
uv run rag-cli data ingest --source /path/to/documents

# 설정 확인
uv run rag-cli config show
```

## 🧪 테스트

### 단위 테스트
```bash
pytest tests/unit/
```

### 통합 테스트
```bash
pytest tests/integration/
```

### API 테스트
```bash
python test_api_endpoints.py
```

## 📊 모니터링

### 메트릭 수집
- API 응답 시간
- 쿼리 처리 통계
- 임베딩 생성 시간
- 벡터 검색 성능
- LLM 응답 시간

### 로그 레벨
- **DEBUG**: 상세 디버깅 정보
- **INFO**: 일반 정보
- **WARNING**: 경고 메시지
- **ERROR**: 오류 정보
- **CRITICAL**: 치명적 오류

## 🔍 주요 기능 상세

### 1. 동적 청킹 (Dynamic Chunking)
- 모델별 최대 토큰 수 고려
- 문맥 유지를 위한 오버랩 설정
- 에피소드 경계 보존

### 2. 배치 임베딩 처리
- 대량 데이터 효율적 처리
- 병렬 처리 지원
- 메모리 최적화

### 3. 다중 모델 앙상블
- 여러 LLM 응답 결합
- 신뢰도 기반 가중치
- 합의 메커니즘

### 4. 캐싱 시스템
- 임베딩 캐시
- 쿼리 결과 캐시
- LLM 응답 캐시

## 📝 개발 가이드라인

### 코드 스타일
- **Black**: 코드 포매팅
- **isort**: import 정렬
- **mypy**: 타입 체킹
- **flake8**: 린팅

### 커밋 컨벤션
- `feat:` 새로운 기능
- `fix:` 버그 수정
- `docs:` 문서 업데이트
- `refactor:` 코드 리팩토링
- `test:` 테스트 추가/수정

## 🎯 프로젝트 목표

1. **확장 가능한 RAG 시스템**: 다양한 LLM과 임베딩 모델 지원
2. **웹소설 특화**: 에피소드 기반 검색 및 처리 최적화
3. **프로덕션 레디**: 인증, 권한, 모니터링 완비
4. **개발자 친화적**: CLI, API, Web UI 제공
5. **성능 최적화**: 배치 처리, 캐싱, 병렬 처리

## 📚 참고 자료

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Milvus 문서](https://milvus.io/docs)
- [LangChain 문서](https://python.langchain.com/)
- [Ollama 문서](https://ollama.ai/)

---

*이 문서는 NovelBot RAG Server의 전체 구조와 주요 기능을 설명합니다. 프로젝트는 지속적으로 개발 중이며, 새로운 기능이 추가될 예정입니다.*