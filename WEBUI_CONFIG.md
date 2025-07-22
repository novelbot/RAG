# WebUI 설정 시스템 가이드

RAG Server의 WebUI는 **하이브리드 설정 시스템**을 사용합니다:

## 설정 계층 구조

```
1. YAML 파일 (webui/settings.yaml) - WebUI 전용 기본 설정
2. 환경 변수 (.env) - 서버 설정과 동기화 및 오버라이드
3. 데이터베이스 (webui/config.db) - 런타임 변경 사항 저장
```

## 설정 분류

### 🔧 **서버 공유 설정** (.env에서 관리)
메인 서버와 WebUI가 공유하는 설정들:

```bash
# .env 파일
SECRET_KEY=your-secret-key        # JWT 토큰 생성
API_HOST=0.0.0.0                 # API 서버 주소
API_PORT=8000                    # API 서버 포트
DEBUG=true                       # 디버그 모드
LOG_LEVEL=INFO                   # 로그 레벨

# LLM 설정
LLM_PROVIDER=ollama              # 현재 사용중인 LLM 제공자
LLM_MODEL=gemma3:27b-it-q8_0     # 현재 모델
OPENAI_API_KEY=sk-...           # API 키들
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### 🎨 **WebUI 전용 설정** (settings.yaml에서 관리)
UI 표시와 사용자 경험 관련 설정들:

```yaml
# webui/settings.yaml
app:
  title: "RAG Server"
  icon: "🤖"
  theme: "light"

ui:
  items_per_page: 20
  enable_dark_mode: true
  show_advanced_options: false

document_categories:
  - name: "Technical"
    color: "#4CAF50"
  - name: "Legal"
    color: "#F44336"

user_roles:
  admin:
    permissions: [read_documents, manage_users, system_config]
```

## 동작 방식

### 1. **시작시 로딩 순서**
1. `settings.yaml`에서 기본값 로드
2. `.env` 파일에서 환경변수 오버라이드 적용
3. 데이터베이스에서 런타임 변경사항 로드

### 2. **LLM 제공자 자동 동기화**
```python
# .env의 LLM 설정이 WebUI에 자동 반영됨
LLM_PROVIDER=ollama  →  웹UI에서 Ollama가 현재 제공자로 표시
LLM_MODEL=gemma3:27b-it-q8_0  →  해당 모델이 기본 모델로 설정

# API 키 존재 여부에 따른 제공자 활성화
OPENAI_API_KEY=sk-... 있음  →  OpenAI 제공자 enabled: true
ANTHROPIC_API_KEY 없음      →  Anthropic 제공자 enabled: false
```

### 3. **환경변수 오버라이드**
WebUI 전용 환경변수로 설정 덮어쓰기:

```bash
# WebUI 전용 오버라이드 환경변수
RAG_APP_TITLE="My Custom RAG"     # 앱 제목 변경
RAG_APP_ICON="🔍"                # 아이콘 변경
RAG_THEME="dark"                 # 테마 변경
RAG_API_TIMEOUT=60               # API 타임아웃 변경
ENABLE_DEMO_USERS=false          # 데모 사용자 비활성화
MAX_UPLOAD_SIZE_MB=200           # 업로드 크기 제한 변경
```

## 설정 관리 방법

### 1. **개발 환경 설정**
```bash
# .env 파일 수정
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
DEBUG=true

# WebUI 전용 설정 (선택사항)
RAG_APP_TITLE="DEV RAG Server"
RAG_THEME="dark"
```

### 2. **프로덕션 환경 설정**
```bash
# .env 파일 (보안 설정)
SECRET_KEY=complex-production-secret
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-production-key
DEBUG=false

# 환경변수로 WebUI 커스터마이징
export RAG_APP_TITLE="Company RAG Server"
export ENABLE_DEMO_USERS=false
export MAX_UPLOAD_SIZE_MB=500
```

### 3. **관리자 UI 사용**
1. WebUI에 admin으로 로그인
2. **Configuration Management** 페이지 접근
3. 실시간으로 설정 변경 (데이터베이스에 저장됨)
4. 변경사항은 즉시 적용, 서버 재시작 불필요

## 설정 우선순위

```
데이터베이스 변경 > 환경변수 > YAML 파일 기본값
```

예시:
```yaml
# settings.yaml
app:
  title: "RAG Server"  # 기본값

# .env
RAG_APP_TITLE="Production RAG"  # 환경변수로 오버라이드

# 관리자 UI에서 변경
app.title = "Custom Company RAG"  # DB에 저장, 최우선 적용
```

## 파일 위치

```
📁 프로젝트 루트/
├── .env                          # 서버 공유 설정
├── webui/
│   ├── settings.yaml            # WebUI 기본 설정
│   ├── config.db               # 런타임 변경사항 (자동생성)
│   ├── config.py               # 설정 로더
│   ├── config_db.py            # DB 설정 관리
│   └── pages/
│       └── config_manager.py   # 관리 UI
```

## 백업 및 복원

### 설정 백업
```python
from webui.config_db import config_db

# 전체 설정 백업
config_db.backup_database("backup/config_20250122.db")

# 특정 카테고리만 내보내기
export_data = config_db.export_config(category="ui")
```

### 설정 복원
```python
# 데이터베이스 복원
config_db.restore_database("backup/config_20250122.db")

# JSON 데이터 가져오기
config_db.import_config(export_data, imported_by="admin")
```

이 구조를 통해 **개발 편의성**과 **운영 유연성**을 모두 제공합니다!