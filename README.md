# NovelBot RAG 서버

웹 소설 콘텐츠 처리 및 시맨틱 검색을 위한 고성능 RAG(Retrieval-Augmented Generation) 서버입니다.

## 주요 기능

- 🚀 Milvus를 사용한 빠른 벡터 검색
- 🤖 다중 LLM 지원 (OpenAI, Anthropic, Ollama)
- 🔐 내장 인증 및 RBAC
- 📊 메트릭 및 모니터링
- 🌊 스트리밍 지원 RESTful API
- 💾 메타데이터 저장을 위한 SQLite 데이터베이스
- 💬 대화 컨텍스트 관리

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- UV 패키지 매니저
- Docker (Milvus용)

### 설치

1. 레포지토리 클론:
```bash
git clone <repository-url>
cd novelbot_RAG_server
```

2. 의존성 설치:
```bash
uv sync
```

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 설정
```

4. 데이터베이스 초기화:
```bash
# 모든 SQLite 데이터베이스 초기화
uv run rag-cli database init --sqlite

# 또는 특정 데이터베이스 초기화
uv run rag-cli database init --sqlite auth metrics
```

5. Milvus(벡터 데이터베이스) 시작:
```bash
docker-compose up -d milvus
```

6. 서버 시작:
```bash
uv run rag-cli serve
```

서버는 `http://localhost:8000`에서 사용할 수 있습니다.

## 데이터베이스 관리

### SQLite 데이터베이스

시스템은 다양한 목적으로 여러 SQLite 데이터베이스를 사용합니다:

- **auth.db**: 사용자 인증 및 권한 부여
- **metrics.db**: 시스템 메트릭 및 성능 추적
- **conversations.db**: 대화 기록 저장
- **user_data.db**: 사용자별 데이터 저장

### GitHub 버전 관리

SQLite 데이터베이스(`.db` 파일)는 설계상 Git에서 추적되지 **않습니다**. 대신:

1. **스키마 파일**은 `database/schemas/`에서 버전 관리됩니다:
   - `auth.sql`: 인증 데이터베이스 스키마
   - `metrics.sql`: 메트릭 추적 스키마
   - `conversations.sql`: 대화 기록 스키마
   - `user_data.sql`: 사용자 데이터 스키마

2. **초기화**: 새로운 개발자는 다음을 실행해야 합니다:
   ```bash
   # 스키마로 모든 데이터베이스 초기화
   uv run rag-cli database init --sqlite
   
   # 또는 독립 실행형 스크립트 사용
   python scripts/init_databases.py
   ```

3. **백업 및 복원**:
   ```bash
   # 데이터베이스 백업
   uv run rag-cli database backup --output backup.sql.gz
   
   # 데이터베이스 복원
   uv run rag-cli database restore --input backup.sql.gz
   ```

### 새로운 개발자를 위한 데이터베이스 설정

1. 레포지토리를 클론한 후:
   ```bash
   # 필요한 모든 데이터베이스 생성
   uv run rag-cli database init --sqlite --force
   ```

2. 이 명령은:
   - 올바른 위치에 데이터베이스 파일 생성
   - SQL 파일에서 스키마 적용
   - 기본 관리자 사용자 생성 (사용자명: `admin`, 비밀번호: `admin123`)
   - 초기 설정 구성

3. **중요**: 첫 로그인 후 기본 관리자 비밀번호를 변경하세요!

### 스키마 업데이트

데이터베이스 스키마를 수정할 때:

1. `database/schemas/`의 스키마 파일 업데이트
2. 필요한 경우 현재 데이터 내보내기:
   ```bash
   sqlite3 auth.db .dump > auth_backup.sql
   ```
3. 새 스키마로 데이터베이스 재초기화:
   ```bash
   uv run rag-cli database init --sqlite auth --force
   ```
4. 필요한 경우 데이터 복원

## API 문서

### 인증

기본 관리자 자격 증명:
- 사용자명: `admin`
- 비밀번호: `admin123`

로그인 엔드포인트:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### 에피소드 처리

```bash
# 에피소드 업로드 및 처리
curl -X POST http://localhost:8000/api/v1/episode/process \
  -H "Authorization: Bearer <token>" \
  -F "file=@episode.txt"
```

### RAG로 채팅

```bash
# 컨텍스트와 함께 채팅
curl -X POST http://localhost:8000/api/v1/episode/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "여기에 질문을 입력하세요",
    "episode_ids": [],
    "novel_ids": []
  }'
```

## CLI 명령어

### 데이터베이스 관리

```bash
# SQLite 데이터베이스 초기화
uv run rag-cli database init --sqlite

# 특정 데이터베이스 초기화
uv run rag-cli database init --sqlite auth metrics

# 강제 재초기화 (기존 데이터 삭제!)
uv run rag-cli database init --sqlite --force

# 데이터베이스 상태
uv run rag-cli database status

# 데이터베이스 연결 테스트
uv run rag-cli database test

# 데이터베이스 백업
uv run rag-cli database backup

# 데이터베이스 복원
uv run rag-cli database restore --input backup.sql.gz
```

### 사용자 관리

```bash
# 사용자 생성
uv run rag-cli user create

# 사용자 목록
uv run rag-cli user list

# 비밀번호 재설정
uv run rag-cli user reset-password --username admin
```

### 데이터 관리

```bash
# 문서 처리
uv run rag-cli data process --file document.txt

# 모든 데이터 삭제
uv run rag-cli data clear --confirm
```

## 설정

### 환경 변수

주요 환경 변수 (`.env.example` 참조):

```env
# LLM 설정
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# 데이터베이스 경로 (선택사항, 기본값 표시)
AUTH_DB_PATH=auth.db
METRICS_DB_PATH=metrics.db
CONVERSATIONS_DB_PATH=data/conversations.db
USER_DATA_DB_PATH=data/user_data.db

# Milvus 설정
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 서버 설정
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

### 설정 파일

`config.json`의 고급 설정:

```json
{
  "database": {
    "driver": "sqlite",
    "name": "auth.db"
  },
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "collection_name": "novelbot_episodes"
  },
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimension": 1536
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7
  }
}
```

## 개발

### 프로젝트 구조

```
novelbot_RAG_server/
├── database/           # 데이터베이스 스키마
│   └── schemas/       # SQL 스키마 파일
├── scripts/           # 유틸리티 스크립트
├── src/              # 소스 코드
│   ├── api/          # API 라우트
│   ├── auth/         # 인증
│   ├── cli/          # CLI 명령어
│   ├── core/         # 핵심 기능
│   ├── database/     # 데이터베이스 유틸리티
│   ├── embedding/    # 임베딩 제공자
│   ├── llm/          # LLM 제공자
│   └── rag/          # RAG 파이프라인
├── data/             # 데이터 디렉토리 (gitignored)
├── *.db              # SQLite 데이터베이스 (gitignored)
└── .env              # 환경 변수 (gitignored)
```

### 테스팅

```bash
# 테스트 실행
uv run pytest

# 커버리지와 함께 실행
uv run pytest --cov=src

# 특정 모듈 테스트
uv run pytest tests/test_auth.py
```

### Docker 배포

```bash
# 이미지 빌드
docker build -t novelbot-rag .

# docker-compose로 실행
docker-compose up -d
```

## 문제 해결

### 데이터베이스 문제

1. **데이터베이스 잠김 오류**:
   - 열려 있는 SQLite 연결을 모두 닫기
   - 서버 재시작

2. **스키마 불일치**:
   ```bash
   # 데이터베이스 재초기화
   uv run rag-cli database init --sqlite --force
   ```

3. **테이블 누락**:
   ```bash
   # 데이터베이스 상태 확인
   uv run rag-cli database status
   
   # 필요한 경우 재초기화
   uv run rag-cli database init --sqlite
   ```

### Milvus 연결 문제

1. Docker가 실행 중인지 확인:
   ```bash
   docker ps
   ```

2. Milvus 시작:
   ```bash
   docker-compose up -d milvus
   ```

3. Milvus 로그 확인:
   ```bash
   docker-compose logs milvus
   ```

## 기여

기여 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 라이선스

[라이선스 정보]