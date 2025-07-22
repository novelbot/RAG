# RAG Server Web UI

사용자 친화적인 웹 인터페이스를 통해 RAG 서버의 모든 기능을 쉽게 사용할 수 있는 Streamlit 기반의 웹 애플리케이션입니다.

## 🌟 주요 기능

### 💻 사용자 인터페이스
- **직관적인 디자인**: 비기술적 사용자도 쉽게 사용할 수 있는 인터페이스
- **반응형 레이아웃**: 데스크톱과 모바일 환경 모두 지원
- **다크/라이트 테마**: 사용자 선호에 따른 테마 선택
- **다국어 지원**: 한국어, 영어 등 다양한 언어 지원

### 🔐 인증 및 권한 관리
- **JWT 기반 인증**: 기존 RAG 서버의 인증 시스템과 연동
- **역할 기반 접근 제어**: User, Manager, Admin 역할별 기능 제한
- **세션 관리**: 자동 로그아웃 및 세션 갱신

### 📊 대시보드
- **시스템 상태 모니터링**: 실시간 서버 상태 및 헬스 체크
- **사용 통계**: 문서 수, 쿼리 수, 활성 사용자 등
- **성능 메트릭**: 쿼리 성공률, 응답 시간, 리소스 사용량
- **최근 활동**: 시스템 내 최근 활동 피드

### 📄 문서 관리
- **파일 업로드**: 드래그 앤 드롭으로 쉬운 파일 업로드
- **배치 업로드**: 여러 파일 동시 업로드 지원
- **처리 상태 추적**: 실시간 문서 처리 상태 모니터링
- **문서 라이브러리**: 검색, 필터링, 분류 기능
- **메타데이터 관리**: 카테고리, 태그, 접근 수준 설정

### 🔍 쿼리 인터페이스
- **RAG 쿼리**: 지식 기반을 활용한 질의응답
- **챗 인터페이스**: LLM과의 직접 대화
- **단일/앙상블 모드**: 빠른 응답 또는 고품질 응답 선택
- **결과 시각화**: 소스 문서, 유사도 점수, 메타데이터 표시
- **쿼리 히스토리**: 이전 쿼리 저장 및 재실행

### 👥 사용자 관리 (관리자용)
- **사용자 계정 관리**: 생성, 수정, 삭제, 활성화/비활성화
- **권한 설정**: 역할별 세부 권한 구성
- **접근 제어**: 부서별, 문서별 접근 권한 설정
- **활동 모니터링**: 사용자별 활동 추적

### ⚙️ 시스템 설정
- **LLM 프로바이더 설정**: OpenAI, Anthropic, Google, Ollama 설정
- **데이터베이스 구성**: 주 데이터베이스 및 벡터 DB 설정
- **시스템 매개변수**: 동시 사용자 수, 쿼리 타임아웃 등
- **모니터링 설정**: 로깅, 메트릭 수집 옵션

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
# uv로 모든 의존성 한 번에 설치 (Web UI 포함)
uv sync
```

### 2. 환경 설정

`.env` 파일에 다음 설정을 추가하세요:

```bash
# Web UI 설정
RAG_API_BASE_URL=http://localhost:8000
RAG_APP_TITLE="RAG Server"
RAG_THEME=light

# 인증 설정
JWT_SECRET_KEY=your-jwt-secret-key
SESSION_TIMEOUT=3600

# 업로드 설정
MAX_UPLOAD_SIZE_MB=100
ALLOWED_FILE_TYPES=txt,pdf,docx,xlsx,md

# 데모 모드 (개발용)
DEMO_MODE=true
ENABLE_DEMO_USERS=true
```

### 3. 웹 UI 실행

```bash
# 간단한 실행
python run_webui.py

# 커스텀 포트 및 호스트
python run_webui.py --host 0.0.0.0 --port 8502

# 개발 모드 (자동 리로드)
python run_webui.py --reload

# 의존성 체크만
python run_webui.py --check-deps
```

### 4. 웹 브라우저에서 접속

```
http://localhost:8501
```

## 🎭 데모 계정

개발 및 테스트용 데모 계정:

### 관리자 계정
- **사용자명**: `admin`
- **비밀번호**: `admin123`
- **권한**: 모든 기능 접근 가능

### 일반 사용자 계정
- **사용자명**: `user`
- **비밀번호**: `user123`
- **권한**: 기본 문서 업로드 및 쿼리 기능

### 매니저 계정
- **사용자명**: `manager`
- **비밀번호**: `manager123`
- **권한**: 사용자 관리를 제외한 대부분 기능

## 📱 페이지 구성

### 🏠 대시보드 (`/`)
- 시스템 상태 개요
- 주요 메트릭 및 통계
- 최근 활동 피드
- 빠른 작업 버튼

### 📄 문서 관리 (`/documents`)
- **업로드 탭**: 파일 업로드 및 메타데이터 설정
- **라이브러리 탭**: 문서 검색, 필터링, 관리
- **처리 상태 탭**: 문서 처리 진행 상황

### 🔍 쿼리 인터페이스 (`/query`)
- **RAG 쿼리 탭**: 지식 기반 질의응답
- **챗 탭**: LLM과의 대화형 인터페이스
- **단일 LLM 탭**: 빠른 단일 모델 응답
- **앙상블 탭**: 다중 모델 고품질 응답

### 👥 관리자 패널 (`/admin`) - 관리자 전용
- **사용자 관리**: 계정 생성, 수정, 삭제
- **시스템 설정**: LLM, DB, 시스템 매개변수
- **모니터링**: 실시간 시스템 상태 및 로그
- **접근 제어**: 역할 및 권한 설정
- **유지보수**: 백업, 정리, 최적화 작업

### ⚙️ 설정 (`/settings`)
- **프로필**: 개인 정보 및 프로필 사진
- **외관**: 테마, 레이아웃, 대시보드 커스터마이징
- **알림**: 이메일 및 앱 내 알림 설정
- **기본 설정**: 쿼리, 문서, 언어 기본값

## 🎨 커스터마이징

### 테마 설정

사용자별로 라이트/다크 테마를 선택할 수 있습니다:

```python
# 설정 페이지에서 테마 변경
st.session_state["theme"] = "dark"  # 또는 "light", "auto"
```

### 대시보드 위젯

대시보드에 표시할 위젯을 사용자가 선택할 수 있습니다:

```python
# 설정에서 대시보드 위젯 구성
dashboard_config = {
    "show_system_status": True,
    "show_recent_queries": True,
    "show_performance_metrics": False,
    # ... 기타 위젯 설정
}
```

## 🔧 개발

### 프로젝트 구조

```
webui/
├── app.py                 # 메인 애플리케이션
├── auth.py                # 인증 관리
├── api_client.py          # API 클라이언트
├── config.py              # 설정 관리
├── __init__.py
└── pages/                 # 페이지 모듈
    ├── __init__.py
    ├── dashboard.py       # 대시보드
    ├── documents.py       # 문서 관리
    ├── query.py           # 쿼리 인터페이스
    ├── admin.py           # 관리자 패널
    └── settings.py        # 설정 페이지
```

### 새 페이지 추가

1. `webui/pages/` 디렉토리에 새 파일 생성
2. `@require_auth` 데코레이터로 인증 보호
3. `show()` 함수로 페이지 내용 구현
4. `app.py`에 라우팅 추가

### API 통합

모든 백엔드 통신은 `api_client.py`를 통해 처리됩니다:

```python
from webui.api_client import get_api_client

api_client = get_api_client()
response = api_client.query_rag("질문", mode="rag", k=5)
```

## 🔒 보안

### 인증 보안
- JWT 토큰 기반 세션 관리
- 자동 토큰 만료 및 갱신
- 안전한 비밀번호 해싱 (프로덕션)

### 데이터 보호
- HTTPS 통신 권장 (프로덕션)
- 민감한 정보 마스킹
- 세션 격리 및 보호

### 접근 제어
- 페이지별 권한 확인
- 역할 기반 기능 제한
- API 엔드포인트 권한 검증

## 📊 모니터링

### 성능 메트릭
- 페이지 로드 시간
- API 응답 시간
- 사용자 세션 통계

### 에러 추적
- 클라이언트 에러 로깅
- API 통신 실패 처리
- 사용자 피드백 수집

## 🚀 배포

### 개발 환경

```bash
# 개발 모드로 실행
python run_webui.py --reload --port 8501
```

### 프로덕션 환경

```bash
# 프로덕션 설정
export ENVIRONMENT=production
export DEMO_MODE=false
export DEBUG=false

# 프로덕션 실행
python run_webui.py --host 0.0.0.0 --port 8501
```

### Docker 배포

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# uv 설치
RUN pip install uv

# 프로젝트 파일 복사 및 의존성 설치
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY webui/ ./webui/
COPY run_webui.py .
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

CMD ["python", "run_webui.py", "--host", "0.0.0.0"]
```

## 🤝 기여하기

1. 이슈 생성 또는 기능 제안
2. 브랜치 생성 (`feature/new-feature`)
3. 변경사항 구현 및 테스트
4. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🆘 지원

### 문제 해결

**일반적인 문제들:**

1. **포트 충돌**: `--port` 옵션으로 다른 포트 사용
2. **의존성 오류**: `uv sync` 재실행
3. **API 연결 실패**: RAG 서버가 실행 중인지 확인

**로그 확인:**
```bash
# 디버그 모드로 실행
export DEBUG=true
python run_webui.py
```

### 연락처

- **문제 보고**: GitHub Issues
- **기능 요청**: GitHub Discussions
- **기술 지원**: team@ragserver.com

---

**RAG Server Web UI**로 강력한 RAG 시스템을 누구나 쉽게 사용할 수 있습니다! 🚀