# Docker Deployment Guide for RAG Server

이 가이드는 RAG Server를 Docker를 사용하여 배포하는 방법을 설명합니다.

## 📋 사전 요구사항

- Docker (20.10+)
- Docker Compose (2.0+)
- 최소 8GB RAM 권장
- 최소 10GB 디스크 공간

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필요에 따라 환경 변수 수정
nano .env
```

### 2. 프로덕션 배포

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f rag-server
```

### 3. 개발 환경

```bash
# 개발 환경 시작
docker-compose -f docker-compose.dev.yml up -d

# 개발 서버 로그 확인
docker-compose -f docker-compose.dev.yml logs -f rag-server-dev
```

## 🏗️ 아키텍처

### 서비스 구성

| 서비스 | 포트 | 설명 |
|--------|------|------|
| rag-server | 8000 | FastAPI RAG 서버 |
| mysql | 3306 | 관계형 데이터베이스 |
| milvus | 19530 | 벡터 데이터베이스 |
| minio | 9000, 9001 | Milvus용 오브젝트 스토리지 |
| etcd | 2379 | Milvus용 메타데이터 저장소 |
| ollama | 11434 | 로컬 LLM (선택사항) |
| webui | 8501 | Streamlit 웹 UI (선택사항) |

### 네트워크

모든 서비스는 `rag-network` 브리지 네트워크에서 통신합니다.

## 🔧 설정 가이드

### 환경 변수 설정

주요 환경 변수들을 `.env` 파일에서 설정하세요:

```bash
# 데이터베이스 설정
DB_HOST=mysql
DB_PORT=3306
DB_NAME=novelbot
DB_USER=raguser
DB_PASSWORD=ragpass

# Milvus 설정
MILVUS_HOST=milvus
MILVUS_PORT=19530

# LLM 설정 (Ollama 사용 시)
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:27b-it-q8_0
OLLAMA_BASE_URL=http://ollama:11434

# 임베딩 설정 (Ollama 사용 시)
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=jeffh/intfloat-multilingual-e5-large-instruct:f32
```

### 데이터 볼륨

영구 데이터는 Docker 볼륨에 저장됩니다:
- `mysql_data`: MySQL 데이터
- `milvus_data`: Milvus 벡터 데이터
- `etcd_data`: Etcd 메타데이터
- `minio_data`: MinIO 오브젝트 저장소
- `ollama_data`: Ollama 모델 데이터

## 🎯 배포 시나리오

### 시나리오 1: 완전 로컬 (추천)

```bash
# Ollama와 로컬 임베딩 사용
docker-compose up -d

# Ollama에 모델 다운로드
docker exec -it rag-ollama ollama pull gemma3:27b-it-q8_0
docker exec -it rag-ollama ollama pull jeffh/intfloat-multilingual-e5-large-instruct:f32
```

### 시나리오 2: 외부 LLM API 사용

```bash
# .env에서 API 키 설정
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-api-key
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your-openai-api-key

# Ollama 없이 시작
docker-compose up -d rag-server mysql milvus etcd minio
```

### 시나리오 3: GPU 가속 (NVIDIA GPU)

```bash
# docker-compose.yml에서 GPU 설정 주석 해제
nano docker-compose.yml

# GPU 지원으로 시작
docker-compose up -d
```

## 📊 모니터링 및 관리

### 헬스 체크

```bash
# 모든 서비스 상태 확인
docker-compose ps

# RAG 서버 헬스 체크
curl http://localhost:8000/health

# Milvus 상태 확인
curl http://localhost:9091/healthz
```

### 로그 확인

```bash
# 모든 서비스 로그
docker-compose logs

# 특정 서비스 로그
docker-compose logs rag-server
docker-compose logs -f mysql

# 실시간 로그 스트림
docker-compose logs -f --tail=100
```

### 데이터베이스 접근

```bash
# MySQL 접근
docker exec -it rag-mysql mysql -u raguser -p novelbot

# Milvus 관리 도구 (Attu) 실행
docker run -p 3000:3000 -e MILVUS_URL=milvus:19530 zilliz/attu:latest
```

## 🔄 업데이트 및 백업

### 애플리케이션 업데이트

```bash
# 코드 변경 후 이미지 재빌드
docker-compose build rag-server

# 서비스 재시작
docker-compose up -d rag-server
```

### 데이터 백업

```bash
# MySQL 백업
docker exec rag-mysql mysqldump -u raguser -p novelbot > backup.sql

# 볼륨 백업
docker run --rm -v mysql_data:/data -v $(pwd):/backup alpine tar czf /backup/mysql_backup.tar.gz /data
```

### 데이터 복원

```bash
# MySQL 복원
docker exec -i rag-mysql mysql -u raguser -p novelbot < backup.sql

# 볼륨 복원
docker run --rm -v mysql_data:/data -v $(pwd):/backup alpine tar xzf /backup/mysql_backup.tar.gz -C /
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. 메모리 부족
```bash
# 메모리 사용량 확인
docker stats

# 불필요한 컨테이너/이미지 정리
docker system prune -a
```

#### 2. 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :8000

# docker-compose.yml에서 포트 변경
ports:
  - "8001:8000"  # 외부 포트 변경
```

#### 3. 데이터베이스 연결 실패
```bash
# 데이터베이스 로그 확인
docker-compose logs mysql

# 연결 테스트
docker exec rag-server python -c "from src.database.base import DatabaseFactory; print('DB OK')"
```

#### 4. Milvus 연결 실패
```bash
# Milvus 헬스 체크
curl http://localhost:9091/healthz

# Milvus 의존성 확인
docker-compose logs etcd minio milvus
```

### 성능 최적화

#### 리소스 제한 설정

```yaml
# docker-compose.yml에 추가
services:
  rag-server:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

#### 프로덕션 설정

```bash
# 프로덕션 환경 변수
echo "APP_ENV=production" >> .env
echo "DEBUG=false" >> .env

# 로그 레벨 조정
echo "LOG_LEVEL=WARNING" >> .env
```

## 🔐 보안 고려사항

### 1. 비밀번호 변경
```bash
# .env에서 기본 비밀번호 변경
DB_PASSWORD=your-secure-password
MYSQL_ROOT_PASSWORD=your-root-password
SECRET_KEY=your-secret-key
```

### 2. 네트워크 보안
```bash
# 외부 접근 제한 (프로덕션)
# docker-compose.yml에서 ports 섹션 제거하고 리버스 프록시 사용
```

### 3. SSL/TLS 설정
```bash
# Nginx 또는 Traefik을 사용한 SSL 종료 추천
```

## 📖 추가 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 가이드](https://docs.docker.com/compose/)
- [Milvus 설치 가이드](https://milvus.io/docs/install_standalone-docker.md)
- [Ollama Docker 가이드](https://hub.docker.com/r/ollama/ollama)

## 🆘 지원

문제가 발생하거나 도움이 필요한 경우:
1. 로그를 확인하세요: `docker-compose logs`
2. 헬스 체크를 실행하세요: `curl http://localhost:8000/health`
3. GitHub Issues에 문제를 보고하세요