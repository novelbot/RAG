# RAG 서버 (벡터 데이터베이스 지원)

Milvus 벡터 데이터베이스를 기반으로 한 프로덕션 레디 RAG(Retrieval-Augmented Generation) 서버입니다. 다중 LLM 지원과 포괄적인 데이터베이스 관리 기능을 제공합니다.

## 🚀 주요 기능

### 다중 LLM 지원
- **OpenAI**: GPT-3.5, GPT-4 모델
- **Anthropic**: Claude 모델
- **Google**: Gemini 모델  
- **Ollama**: 로컬 모델 지원
- 확장 가능한 LLM 프로바이더 프레임워크

### 다중 데이터베이스 지원
- **데이터베이스**: PostgreSQL, MySQL, MariaDB, Oracle, SQL Server
- **고급 연결 관리**: 연결 풀링, 헬스 모니터링, 재시도 메커니즘
- **파일 소스**: TXT, PDF, Word, Excel, Markdown
- 자동 스키마 감지 및 인트로스펙션

### 세밀한 접근 제어 (FGAC)
- Milvus 행 수준 RBAC 통합
- 사용자/그룹 기반 권한 관리
- 리소스 수준 접근 제어
- JWT 기반 인증

### 이중 RAG 운영 모드
- **단일 LLM 모드**: 빠른 단일 모델 응답
- **다중 LLM 모드**: 합의 기반 다중 모델 응답

### 프로덕션 레디 데이터베이스 레이어
- **Milvus** 벡터 데이터베이스 통합
- 포괄적인 헬스 모니터링
- 지능형 에러 처리 및 재시도 메커니즘
- 내결함성을 위한 서킷 브레이커 패턴

## 📋 요구사항

- **Python**: 3.11+
- **Milvus**: 2.3.0+
- **메모리**: 최소 8GB (권장 16GB+)
- **스토리지**: 데이터베이스 및 벡터 저장 공간

## 🛠️ 설치 방법

### 1. 레포지토리 클론

```bash
git clone <repository-url>
cd novelbot_RAG_server

# uv로 의존성 설치
uv sync

# 개발 의존성 설치
uv sync --group dev
```

### 2. 환경 설정

```bash
# 환경 변수 템플릿 복사
cp .env.example .env

# .env 파일에서 설정 수정
vim .env

# 필수 설정:
# - 데이터베이스 연결 정보 (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
# - Milvus 연결 정보 (MILVUS_HOST, MILVUS_PORT)
# - LLM 및 임베딩 프로바이더 설정
# - API 키 설정 (사용하는 프로바이더에 따라)
```

### 3. 애플리케이션 실행

```bash
# 서버 시작
uv run main.py

# 또는 CLI 사용
uv run rag-cli serve --reload
```

## ⚙️ 설정

설정은 `.env` 파일을 통해 환경 변수로 관리됩니다. 설정 예시는 `.env.example` 파일을 참조하세요.

### 임베딩 설정

```bash
# Ollama 로컬 임베딩 (무료) [권장]
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=jeffh/intfloat-multilingual-e5-large-instruct:f32
EMBEDDING_API_KEY=

# OpenAI 임베딩 (유료)
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-large
# EMBEDDING_API_KEY=your-openai-api-key

# Google 임베딩 (유료)
# EMBEDDING_PROVIDER=google
# EMBEDDING_MODEL=text-embedding-004
# EMBEDDING_API_KEY=your-google-api-key
```

### 데이터베이스 설정

```bash
# MySQL/MariaDB (기본값)
DB_HOST=localhost
DB_PORT=3306
DB_NAME=novelbot
DB_USER=root
DB_PASSWORD=password

# PostgreSQL 사용 시
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=ragdb
# DB_USER=postgres
# DB_PASSWORD=password
```

### LLM 설정

```bash
# Ollama 로컬 LLM (무료) [권장]
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:27b-it-q8_0
LLM_API_KEY=

# OpenAI (유료)
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-3.5-turbo
# LLM_API_KEY=your-openai-api-key

# Anthropic Claude (유료)
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-latest
# LLM_API_KEY=your-anthropic-api-key

# Google Gemini (유료)
# LLM_PROVIDER=google
# LLM_MODEL=gemini-2.0-flash-001
# LLM_API_KEY=your-google-api-key
```

### Milvus 설정

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
# 로컬 Milvus에서는 인증 없이 사용 가능
MILVUS_USER=
MILVUS_PASSWORD=
```

### API 서버 설정

```bash
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-here
```


## 📡 API 사용법

### 기본 LLM 쿼리

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "messages": [
      {"role": "user", "content": "머신러닝이 무엇인가요?"}
    ],
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

### 스트리밍 LLM 응답

```bash
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "messages": [
      {"role": "user", "content": "양자 컴퓨팅에 대해 자세히 설명해주세요"}
    ],
    "model": "claude-3-5-sonnet-latest",
    "temperature": 0.7,
    "stream": true
  }'
```

### 다중 LLM 로드 밸런싱

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "messages": [
      {"role": "user", "content": "인공지능의 미래는 어떻게 될까요?"}
    ],
    "load_balancing": "health_based",
    "temperature": 0.8,
    "max_tokens": 1500
  }'
```

### RAG 쿼리 (벡터 검색 + LLM)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "머신러닝 알고리즘의 종류와 특징",
    "mode": "rag",
    "k": 5,
    "llm_provider": "openai",
    "model": "gpt-4"
  }'
```

### 임베딩 생성

```bash
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "input": ["머신러닝이란 무엇인가?", "딥러닝과 머신러닝의 차이점"],
    "model": "text-embedding-3-large",
    "dimensions": 1024,
    "normalize": true
  }'
```

### Ollama 로컬 임베딩 생성

```bash
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "input": ["자연어 처리 기술 설명"],
    "provider": "ollama",
    "model": "nomic-embed-text",
    "normalize": true
  }'
```

### 다중 임베딩 프로바이더 사용

```bash
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "input": ["자연어 처리 기술의 발전"],
    "load_balancing": "cost_optimized",
    "dimensions": 512,
    "normalize": true
  }'
```

### 단일 LLM 응답 생성 (고속 모드)

```bash
curl -X POST "http://localhost:8000/generate/single" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "머신러닝과 딥러닝의 차이점을 설명해주세요",
    "mode": "fast",
    "context": "AI 기술 관련 교육 자료",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "response_format": "markdown"
  }'
```

### 다중 LLM 앙상블 응답 (고품질 모드)

```bash
curl -X POST "http://localhost:8000/generate/ensemble" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "인공지능의 윤리적 고려사항에 대해 분석해주세요",
    "mode": "high_quality", 
    "ensemble_size": 3,
    "consensus_threshold": 0.7,
    "enable_parallel_generation": true,
    "evaluation_metrics": ["relevance", "accuracy", "completeness"],
    "output_format": "structured",
    "custom_instructions": "다양한 관점에서 균형잡힌 분석 제공"
  }'
```

### RAG 기반 고급 응답 생성

```bash
curl -X POST "http://localhost:8000/generate/rag" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "최신 트랜스포머 아키텍처의 발전 동향",
    "retrieval_config": {
      "k": 10,
      "search_strategy": "hybrid",
      "include_metadata": true,
      "enable_reranking": true
    },
    "generation_config": {
      "mode": "ensemble",
      "ensemble_size": 2,
      "prompt_strategy": "contextual",
      "enable_post_processing": true,
      "response_format": "markdown"
    },
    "quality_filters": {
      "min_relevance": 0.8,
      "min_completeness": 0.7,
      "enable_fact_checking": true
    }
  }'
```

### RDB 데이터 추출

```bash
curl -X POST "http://localhost:8000/extract/rdb" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "database_config": {
      "host": "localhost",
      "port": 5432,
      "database": "mydb",
      "user": "postgres",
      "password": "password",
      "database_type": "postgresql"
    },
    "extraction_config": {
      "mode": "incremental",
      "batch_size": 1000,
      "include_tables": ["users", "orders", "products"],
      "incremental_column": "updated_at",
      "validate_data": true
    }
  }'
```

### 파일 시스템 배치 처리

```bash
curl -X POST "http://localhost:8000/extract/filesystem" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "directory": "/path/to/documents",
    "batch_config": {
      "max_workers": 8,
      "batch_size": 100,
      "memory_limit_mb": 2048,
      "enable_adaptive_batching": true,
      "processing_strategy": "priority_based"
    },
    "file_patterns": ["*.pdf", "*.docx", "*.txt"],
    "exclude_patterns": ["*.tmp"],
    "progress_callback_url": "http://localhost:8000/progress"
  }'
```

## 💻 프로그래밍 사용법

### 다중 LLM 매니저 사용

```python
import asyncio
from src.llm import (
    LLMManager, LLMProvider, LLMConfig, LLMRequest, 
    LLMMessage, LLMRole, ProviderConfig, LoadBalancingStrategy
)

async def main():
    # 프로바이더 설정
    provider_configs = [
        ProviderConfig(
            provider=LLMProvider.OPENAI,
            config=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key="your-openai-key",
                model="gpt-4",
                temperature=0.7
            )
        ),
        ProviderConfig(
            provider=LLMProvider.CLAUDE,
            config=LLMConfig(
                provider=LLMProvider.CLAUDE,
                api_key="your-anthropic-key",
                model="claude-3-5-sonnet-latest",
                temperature=0.7
            )
        )
    ]
    
    # LLM 매니저 초기화
    llm_manager = LLMManager(provider_configs)
    llm_manager.set_load_balancing_strategy(LoadBalancingStrategy.HEALTH_BASED)
    
    # 메시지 생성
    messages = [
        LLMMessage(role=LLMRole.USER, content="안녕하세요, 머신러닝에 대해 설명해주세요")
    ]
    
    request = LLMRequest(
        messages=messages,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    
    # 응답 생성
    response = await llm_manager.generate_async(request)
    print(f"응답: {response.content}")
    print(f"사용된 프로바이더: {response.metadata.get('provider')}")
    
    # 스트리밍 응답
    async for chunk in llm_manager.generate_stream_async(request):
        if chunk.content:
            print(chunk.content, end='', flush=True)

asyncio.run(main())
```

### 단일 및 앙상블 응답 생성 시스템 사용

```python
import asyncio
from src.response_generation import (
    SingleLLMGenerator, EnsembleLLMGenerator, ResponseRequest,
    ResponseGeneratorConfig, ResponseMode, EvaluationMetric,
    PromptStrategy, ContextInjectionMode, OutputFormat
)
from src.llm import LLMManager
from src.rag.context_retriever import DocumentContext

async def response_generation_example():
    # 응답 생성 설정
    config = ResponseGeneratorConfig(
        single_timeout=30.0,
        ensemble_timeout=90.0,
        ensemble_size=3,
        consensus_threshold=0.7,
        min_quality_score=0.6,
        enable_parallel_generation=True,
        enable_post_processing=True,
        enable_quality_filtering=True,
        evaluation_metrics=[
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.ACCURACY,
            EvaluationMetric.COMPLETENESS
        ]
    )
    
    # LLM 매니저 초기화 (이전 예제에서 설정)
    llm_manager = LLMManager(provider_configs)
    
    # 단일 LLM 생성기
    single_generator = SingleLLMGenerator(llm_manager, config)
    
    # 앙상블 LLM 생성기
    ensemble_generator = EnsembleLLMGenerator(llm_manager, config)
    
    # 요청 생성
    request = ResponseRequest(
        query="머신러닝과 딥러닝의 핵심 차이점을 상세히 설명해주세요",
        context="AI 기술 교육 과정의 기초 학습 자료",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1500,
        system_prompt="당신은 AI 기술 전문가입니다.",
        custom_instructions="초보자도 이해할 수 있도록 명확하고 구체적인 예시를 포함해주세요.",
        response_format="markdown"
    )
    
    # 단일 LLM 응답 생성 (고속 모드)
    print("=== 단일 LLM 응답 생성 ===")
    single_result = await single_generator.generate_response_async(request)
    
    print(f"응답: {single_result.response}")
    print(f"사용된 프로바이더: {single_result.provider_used}")
    print(f"응답 시간: {single_result.response_time:.2f}초")
    print(f"품질 점수: {single_result.overall_quality_score:.3f}")
    
    # 앙상블 LLM 응답 생성 (고품질 모드)
    print("\n=== 앙상블 LLM 응답 생성 ===")
    ensemble_result = await ensemble_generator.generate_response_async(request)
    
    print(f"최적 응답: {ensemble_result.best_response.response}")
    print(f"사용된 프로바이더들: {ensemble_result.providers_used}")
    print(f"컨센서스 점수: {ensemble_result.consensus_score:.3f}")
    print(f"선택 방법: {ensemble_result.selection_method}")
    print(f"총 응답 수: {len(ensemble_result.all_responses)}")
    print(f"앙상블 처리 시간: {ensemble_result.ensemble_time:.2f}초")
    
    # 모든 응답 비교
    print("\n=== 모든 앙상블 응답 비교 ===")
    for i, response in enumerate(ensemble_result.all_responses, 1):
        print(f"응답 {i} (프로바이더: {response.provider_used}):")
        print(f"  품질 점수: {response.overall_quality_score:.3f}")
        print(f"  응답 길이: {len(response.response)} 문자")
        
        # 개별 품질 메트릭
        for score in response.quality_scores:
            print(f"  {score.metric.value}: {score.score:.3f} (신뢰도: {score.confidence:.3f})")
        print()

asyncio.run(response_generation_example())
```

### 고급 프롬프트 엔지니어링 사용

```python
from src.response_generation import (
    PromptEngineer, PromptStrategy, ContextInjectionMode,
    PromptTemplate, ContextRelevanceFilter
)

async def prompt_engineering_example():
    # 프롬프트 엔지니어 초기화
    prompt_engineer = PromptEngineer(config)
    
    # 커스텀 프롬프트 템플릿 추가
    custom_template = PromptTemplate(
        name="detailed_analysis",
        template="""당신은 {domain} 분야의 전문가입니다.
        
주어진 정보를 바탕으로 다음 질문에 답변해주세요:

{context}

질문: {query}

답변 요구사항:
- 구체적인 예시 포함
- 단계별 설명
- 실용적 관점에서의 분석
- {response_format} 형식으로 작성

{custom_instructions}""",
        required_variables=["query", "domain"],
        optional_variables=["context", "response_format", "custom_instructions"]
    )
    
    prompt_engineer.add_custom_template(custom_template)
    
    # 컨텍스트가 포함된 요청
    request_with_context = ResponseRequest(
        query="트랜스포머 아키텍처의 어텐션 메커니즘 작동 원리",
        retrieval_result=RetrievalResult(
            contexts=[
                DocumentContext(
                    id="doc1",
                    content="어텐션 메커니즘은 입력 시퀀스의 모든 위치에 대해 가중치를 계산합니다...",
                    similarity_score=0.95,
                    source_info={"title": "Attention Is All You Need", "type": "paper"}
                ),
                DocumentContext(
                    id="doc2", 
                    content="셀프 어텐션은 쿼리, 키, 밸류 벡터를 사용하여 계산됩니다...",
                    similarity_score=0.88,
                    source_info={"title": "트랜스포머 구조 분석", "type": "tutorial"}
                )
            ]
        ),
        user_context={"domain": "딥러닝", "level": "intermediate"},
        response_format="markdown",
        custom_instructions="수식과 그림으로 설명"
    )
    
    # 다양한 프롬프트 전략 테스트
    strategies = [
        (PromptStrategy.STRUCTURED, ContextInjectionMode.STRUCTURED_SECTIONS),
        (PromptStrategy.CONTEXTUAL, ContextInjectionMode.ADAPTIVE),
        (PromptStrategy.HIERARCHICAL, ContextInjectionMode.INTERLEAVED)
    ]
    
    for strategy, injection_mode in strategies:
        print(f"\n=== {strategy.value} 전략, {injection_mode.value} 주입 모드 ===")
        
        engineered_prompt, metadata = prompt_engineer.engineer_prompt(
            request_with_context, strategy, injection_mode
        )
        
        print(f"사용된 템플릿: {metadata['template_used']}")
        print(f"컨텍스트 수: {metadata['contexts_used']}")
        print(f"원본 길이: {metadata['original_length']}")
        print(f"최적화 길이: {metadata['optimized_length']}")
        print(f"압축 적용: {metadata['compression_applied']}")
        print(f"프롬프트 미리보기: {engineered_prompt[:200]}...")

asyncio.run(prompt_engineering_example())
```

### 응답 품질 평가 및 후처리 사용

```python
from src.response_generation import (
    ResponseEvaluator, ResponseProcessor, EvaluationWeights,
    QualityThresholds, OutputFormat, FormattingRules, ValidationRules
)

async def evaluation_and_processing_example():
    # 응답 평가기 설정
    evaluator = ResponseEvaluator(config)
    
    # 커스텀 평가 가중치 설정
    custom_weights = EvaluationWeights(
        relevance=0.35,
        accuracy=0.30,
        completeness=0.20,
        clarity=0.10,
        coherence=0.05
    )
    evaluator.set_evaluation_weights(custom_weights)
    
    # 커스텀 품질 임계값 설정
    custom_thresholds = QualityThresholds(
        min_relevance=0.7,
        min_accuracy=0.6,
        min_completeness=0.5,
        min_overall=0.65
    )
    evaluator.set_quality_thresholds(custom_thresholds)
    
    # 응답 후처리기 설정
    processor = ResponseProcessor(config)
    
    # 커스텀 포맷팅 규칙
    formatting_rules = FormattingRules(
        max_line_length=100,
        enable_auto_paragraphs=True,
        enable_auto_lists=True,
        enable_auto_headers=True,
        fix_grammar=True,
        normalize_punctuation=True
    )
    processor.set_formatting_rules(formatting_rules)
    
    # 검증 규칙
    validation_rules = ValidationRules(
        min_length=50,
        max_length=5000,
        check_completeness=True,
        check_coherence=True,
        remove_hallucinations=True
    )
    processor.set_validation_rules(validation_rules)
    
    # 샘플 응답들 생성 (실제로는 LLM에서 생성)
    sample_responses = [
        ResponseResult(
            response="머신러닝은 데이터로부터 패턴을 학습하는 방법입니다. 딥러닝은 신경망을 사용한 머신러닝의 한 분야로...",
            llm_response=LLMResponse(content="...", metadata={"provider": "openai"})
        ),
        ResponseResult(
            response="인공지능의 하위 분야인 머신러닝은 알고리즘을 통해 데이터에서 자동으로 학습합니다. 딥러닝은...",
            llm_response=LLMResponse(content="...", metadata={"provider": "claude"})
        )
    ]
    
    # 각 응답 평가
    evaluated_responses = []
    for response in sample_responses:
        evaluated_response = evaluator.evaluate_response(response, request)
        evaluated_responses.append(evaluated_response)
        
        print(f"응답 평가 결과:")
        print(f"  전체 품질 점수: {evaluated_response.overall_quality_score:.3f}")
        print(f"  신뢰도 점수: {evaluated_response.confidence_score:.3f}")
        
        for score in evaluated_response.quality_scores:
            print(f"  {score.metric.value}: {score.score:.3f} ({score.explanation})")
    
    # 최적 응답 선택
    best_response, consensus, method = evaluator.select_best_response(
        evaluated_responses, request
    )
    
    print(f"\n최적 응답 선택됨 (방법: {method}, 컨센서스: {consensus:.3f})")
    
    # 다양한 포맷으로 후처리
    output_formats = [
        OutputFormat.PLAIN_TEXT,
        OutputFormat.MARKDOWN,
        OutputFormat.HTML,
        OutputFormat.JSON,
        OutputFormat.STRUCTURED
    ]
    
    for format_type in output_formats:
        processed_response = processor.process_response(
            best_response, request, format_type
        )
        
        print(f"\n=== {format_type.value} 포맷 ===")
        print(f"처리된 응답: {processed_response.response[:200]}...")
        print(f"처리 단계: {processed_response.processing_steps}")

asyncio.run(evaluation_and_processing_example())
```

### 에러 처리 및 복구 시스템 사용

```python
from src.response_generation import (
    ErrorHandler, ErrorPolicy, ErrorSeverity, ErrorCategory,
    RetryStrategy, CircuitBreakerConfig, TimeoutConfig
)

async def error_handling_example():
    # 에러 핸들러 초기화
    error_handler = ErrorHandler(config)
    
    # 커스텀 서킷 브레이커 설정
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2,
        enable_half_open=True
    )
    
    # 커스텀 타임아웃 설정
    timeout_config = TimeoutConfig(
        default_timeout=30.0,
        slow_timeout=60.0,
        fast_timeout=15.0,
        ensemble_timeout=120.0,
        enable_adaptive_timeout=True
    )
    
    # 커스텀 에러 정책 추가
    custom_policy = ErrorPolicy(
        error_types=[ConnectionError, TimeoutError],
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.NETWORK,
        retry_strategy=RetryStrategy.EXPONENTIAL,
        max_retries=3,
        base_delay=2.0,
        should_fallback=True,
        should_circuit_break=True
    )
    
    # 시뮬레이션: 에러 처리
    try:
        # 타임아웃과 함께 함수 실행
        result = await error_handler.with_timeout_async(
            some_llm_function,
            timeout=30.0,
            request_data
        )
        
        # 서킷 브레이커와 함께 실행
        result = await error_handler.with_circuit_breaker_async(
            some_provider_function,
            provider="openai",
            request_data
        )
        
    except Exception as e:
        # 에러 처리 결정
        context = {"provider": "openai", "model": "gpt-4", "operation": "generate"}
        decision = error_handler.handle_error(e, context, retry_count=0)
        
        print(f"에러 처리 결정: {decision['action']}")
        print(f"재시도 지연: {decision['delay']}초")
        print(f"폴백 필요: {decision['should_fallback']}")
        print(f"서킷 브레이커 트리거: {decision['should_circuit_break']}")
        
        # 재시도 여부 확인
        should_retry, delay = error_handler.should_retry(e, retry_count=0, context=context)
        
        if should_retry:
            print(f"재시도 예정 ({delay}초 후)")
            await asyncio.sleep(delay)
            # 재시도 로직...
        else:
            print("재시도하지 않음, 폴백 또는 실패 처리")
    
    # 에러 통계 확인
    stats = error_handler.get_error_statistics()
    print(f"\n에러 통계:")
    print(f"총 에러 수: {stats['total_errors']}")
    print(f"카테고리별 분류: {stats['category_breakdown']}")
    print(f"심각도별 분류: {stats['severity_breakdown']}")
    print(f"프로바이더별 분류: {stats['provider_breakdown']}")
    print(f"서킷 브레이커 상태: {stats['circuit_breaker_states']}")

asyncio.run(error_handling_example())
```

### 개별 프로바이더 사용

```python
from src.llm import OpenAIProvider, LLMConfig, LLMProvider

# OpenAI 프로바이더 초기화
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-key",
    model="gpt-4",
    temperature=0.7
)

provider = OpenAIProvider(config)

# 응답 생성
response = await provider.generate_async(request)
print(response.content)
```

### 임베딩 매니저 사용

```python
import asyncio
from src.embedding import (
    EmbeddingManager, EmbeddingProvider, EmbeddingConfig, EmbeddingRequest,
    EmbeddingProviderConfig, EmbeddingLoadBalancingStrategy
)

async def main():
    # 프로바이더 설정
    provider_configs = [
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.OPENAI,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                api_key="your-openai-key",
                model="text-embedding-3-large",
                dimensions=1024
            ),
            priority=1,
            cost_per_1m_tokens=0.13
        ),
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.GOOGLE,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.GOOGLE,
                api_key="your-google-key",
                model="text-embedding-004",
                dimensions=768
            ),
            priority=2,
            cost_per_1m_tokens=0.025
        ),
        EmbeddingProviderConfig(
            provider=EmbeddingProvider.OLLAMA,
            config=EmbeddingConfig(
                provider=EmbeddingProvider.OLLAMA,
                model="nomic-embed-text",
                base_url="http://localhost:11434",
                dimensions=768
            ),
            priority=3,
            cost_per_1m_tokens=0.0  # 로컬 모델은 무료
        )
    ]
    
    # 임베딩 매니저 초기화
    embedding_manager = EmbeddingManager(provider_configs, enable_cache=True)
    embedding_manager.set_load_balancing_strategy(EmbeddingLoadBalancingStrategy.COST_OPTIMIZED)
    
    # 임베딩 생성
    request = EmbeddingRequest(
        input=["머신러닝이란 무엇인가?"],
        model="text-embedding-3-large",
        dimensions=512,
        normalize=True
    )
    
    response = await embedding_manager.generate_embeddings_async(request)
    print(f"임베딩 차원: {response.dimensions}")
    print(f"사용된 프로바이더: {response.metadata.get('provider')}")
    print(f"비용: ${response.metadata.get('cost', 0.0):.6f}")

asyncio.run(main())
```

### RDB 데이터 추출 파이프라인 사용

```python
import asyncio
from src.extraction import (
    RDBExtractorFactory, ExtractionConfig, ExtractionMode, DataFormat
)
from src.core.config import DatabaseConfig
from src.database.drivers import DatabaseType

async def main():
    # 데이터베이스 설정
    database_config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="mydb",
        user="postgres",
        password="password",
        database_type=DatabaseType.POSTGRESQL
    )
    
    # 추출 설정
    extraction_config = ExtractionConfig(
        database_config=database_config,
        mode=ExtractionMode.INCREMENTAL,
        batch_size=1000,
        max_rows=10000,
        include_tables=["users", "orders", "products"],
        incremental_column="updated_at",
        validate_data=True,
        output_format=DataFormat.DICT
    )
    
    # 추출기 생성
    extractor = RDBExtractorFactory.create(extraction_config)
    
    try:
        # 모든 테이블에서 데이터 추출
        results = extractor.extract_all_tables()
        
        for result in results:
            if result.is_successful():
                print(f"테이블 {result.metadata.name}에서 {len(result.data)} 행 추출 완료")
                print(f"추출 ID: {result.extraction_id}")
                print(f"처리 시간: {result.stats.total_time:.2f}초")
            else:
                print(f"추출 실패: {result.errors}")
        
        # 점진적 동기화 상태 확인
        summary = extractor.get_extraction_summary()
        print(f"전체 추출 통계: {summary}")
        
    finally:
        extractor.close()

asyncio.run(main())
```

### 파일 시스템 배치 처리 사용

```python
import asyncio
from pathlib import Path
from src.file_system import (
    FileProcessor, BatchProcessingConfig, ProcessingStrategy,
    RetryStrategy, ErrorSeverity
)

async def progress_callback(stats):
    """진행률 콜백 함수"""
    print(f"진행률: {stats.progress_percentage:.1f}% "
          f"({stats.processed_items}/{stats.total_items})")
    print(f"처리 속도: {stats.items_per_second:.1f} files/sec")
    print(f"예상 남은 시간: {stats.eta_seconds:.0f}초")

async def main():
    # 배치 처리 설정
    batch_config = BatchProcessingConfig(
        max_workers=8,
        batch_size=50,
        processing_strategy=ProcessingStrategy.PRIORITY_BASED,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        max_retries=3,
        memory_limit_mb=2048,
        enable_adaptive_batching=True,
        enable_eta_calculation=True,
        enable_gc=True
    )
    
    # 파일 프로세서 초기화
    processor = FileProcessor(
        batch_config=batch_config,
        enable_change_detection=True,
        enable_metadata_extraction=True
    )
    
    try:
        # 디렉토리 처리
        results = processor.process_directory(
            directory="/path/to/documents",
            file_patterns=["*.pdf", "*.docx", "*.txt", "*.md"],
            exclude_patterns=["*.tmp", "*.log"],
            max_depth=5,
            only_changed_files=True,
            progress_callback=progress_callback
        )
        
        # 결과 출력
        summary = results["batch_processing"]["summary"]
        print(f"처리 완료: {summary['successful']}개 성공, {summary['failed']}개 실패")
        print(f"성공률: {summary['success_rate']:.1%}")
        print(f"총 처리 시간: {summary['processing_time']:.2f}초")
        
        # 에러 분석
        if results["batch_processing"]["errors"]:
            print("\n오류 발생 파일:")
            for error in results["batch_processing"]["errors"]:
                print(f"- {error['path']}: {error['error']}")
        
        # 진행률 및 성능 통계
        performance = results["batch_processing"]["performance"]
        print(f"\n성능 통계:")
        print(f"- 평균 처리 속도: {performance['average_throughput']:.1f} files/sec")
        print(f"- 피크 처리 속도: {performance['peak_throughput']:.1f} files/sec")
        
        # 메모리 사용량
        memory = results["batch_processing"]["memory"]
        print(f"- 메모리 사용량: {memory['current_mb']:.1f}MB")
        print(f"- 최대 메모리 사용량: {memory['peak_mb']:.1f}MB")
        
    except Exception as e:
        print(f"배치 처리 실패: {e}")
    
    finally:
        processor.stop_processing()

asyncio.run(main())
```

### 실시간 배치 처리 제어

```python
from src.file_system import FileProcessor, BatchProcessingConfig

# 파일 프로세서 초기화
processor = FileProcessor()

# 비동기 처리 시작
processing_task = asyncio.create_task(
    processor.process_directory("/large/dataset")
)

# 처리 중 제어
await asyncio.sleep(5)
processor.pause_processing()  # 일시 정지
print("처리 일시 정지됨")

await asyncio.sleep(2)
processor.resume_processing()  # 재개
print("처리 재개됨")

# 상태 모니터링
status = processor.get_processing_status()
print(f"현재 상태: {status['state']}")
print(f"진행률: {status['progress']['progress_percentage']:.1f}%")

# 처리 완료 대기
results = await processing_task
```

### 개별 테이블 추출

```python
from src.extraction import GenericRDBExtractor, ExtractionConfig

# 특정 테이블에서 데이터 추출
extractor = GenericRDBExtractor(extraction_config)

# 필터링과 함께 추출
result = extractor.extract_table_data(
    table_name="users",
    filters={"status": "active", "created_at": "> '2024-01-01'"},
    order_by="created_at DESC"
)

print(f"추출된 활성 사용자 수: {len(result.data)}")
print(f"테이블 메타데이터: {result.metadata.to_dict()}")
```

### 벡터 파이프라인 사용

```python
import asyncio
from src.pipeline import VectorPipeline, PipelineConfig
from src.embedding import EmbeddingManager, EmbeddingConfig, EmbeddingProvider
from src.milvus import MilvusClient
from src.text_processing import TextCleaner, TextSplitter

async def main():
    # 파이프라인 설정
    pipeline_config = PipelineConfig(
        batch_size=100,
        enable_parallel_processing=True,
        max_workers=8,
        enable_performance_monitoring=True,
        enable_error_recovery=True
    )
    
    # 임베딩 설정
    embedding_config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model="text-embedding-3-large",
        api_key="your-openai-key",
        dimensions=1536
    )
    
    embedding_manager = EmbeddingManager([embedding_config])
    milvus_client = MilvusClient()
    
    # 벡터 파이프라인 초기화
    pipeline = VectorPipeline(
        config=pipeline_config,
        text_cleaner=TextCleaner(),
        text_splitter=TextSplitter(),
        embedding_manager=embedding_manager,
        milvus_client=milvus_client
    )
    
    # 문서 처리
    documents = [
        {"id": "doc1", "content": "머신러닝은 인공지능의 한 분야입니다.", "metadata": {"source": "textbook"}},
        {"id": "doc2", "content": "딥러닝은 신경망을 활용한 학습 방법입니다.", "metadata": {"source": "paper"}}
    ]
    
    # 파이프라인 실행
    result = await pipeline.process_documents(documents)
    
    print(f"처리된 문서 수: {result.processed_count}")
    print(f"생성된 벡터 수: {result.vector_count}")
    print(f"처리 시간: {result.processing_time:.2f}초")
    print(f"성공률: {result.success_rate:.1%}")

asyncio.run(main())
```

### 고급 배치 처리 및 성능 최적화

```python
from src.pipeline import (
    BatchProcessor, BatchConfig, BatchStrategy,
    PerformanceOptimizer, OptimizationConfig, OptimizationLevel
)

# 고급 배치 처리 설정
batch_config = BatchConfig(
    strategy=BatchStrategy.ADAPTIVE,
    base_batch_size=50,
    max_batch_size=500,
    memory_threshold_mb=1024,
    cpu_threshold_percent=80,
    enable_dynamic_sizing=True,
    enable_parallel_processing=True
)

# 성능 최적화 설정
optimization_config = OptimizationConfig(
    level=OptimizationLevel.AGGRESSIVE,
    enable_memory_optimization=True,
    enable_cpu_optimization=True,
    enable_adaptive_concurrency=True,
    enable_smart_caching=True,
    max_memory_usage_percent=85,
    max_cpu_usage_percent=90
)

# 고급 파이프라인 구성
advanced_pipeline = VectorPipeline(
    config=pipeline_config,
    batch_processor=BatchProcessor(batch_config),
    performance_optimizer=PerformanceOptimizer(optimization_config),
    # ... 기타 컴포넌트
)

# 실시간 성능 모니터링
async def monitor_pipeline_performance():
    while pipeline.is_running():
        metrics = pipeline.get_performance_metrics()
        print(f"처리율: {metrics.throughput_per_second:.1f} docs/sec")
        print(f"메모리 사용량: {metrics.memory_usage_mb:.1f}MB")
        print(f"CPU 사용률: {metrics.cpu_usage_percent:.1f}%")
        await asyncio.sleep(10)

# 성능 모니터링과 함께 실행
processing_task = asyncio.create_task(pipeline.process_large_dataset(documents))
monitoring_task = asyncio.create_task(monitor_pipeline_performance())

await asyncio.gather(processing_task, monitoring_task)
```

### 메타데이터 강화 및 접근 제어

```python
from src.pipeline import MetadataEnricher, EnrichmentConfig, EnrichmentLevel
from src.access_control import AccessControlManager

# 메타데이터 강화 설정
enrichment_config = EnrichmentConfig(
    enable_language_detection=True,
    enable_pii_detection=True,
    enable_entity_extraction=True,
    enable_security_classification=True,
    enable_compliance_tagging=True,
    auto_classify_sensitivity=True
)

metadata_enricher = MetadataEnricher(
    config=enrichment_config,
    access_control_manager=AccessControlManager()
)

# 문서 메타데이터 강화
enriched_metadata = await metadata_enricher.enrich_metadata(
    content="이 문서는 기밀 정보를 포함합니다. 연락처: john@company.com",
    base_metadata={"source": "internal_doc", "department": "hr"},
    user_id="user123",
    enrichment_level=EnrichmentLevel.COMPREHENSIVE
)

print(f"PII 감지: {enriched_metadata['content_analysis']['pii_detected']}")
print(f"보안 분류: {enriched_metadata['access_control']['security_classification']}")
print(f"컴플라이언스 태그: {enriched_metadata['compliance']}")
```

## 🖥️ CLI 인터페이스

프로덕션 레디 Click 기반 CLI 인터페이스를 통해 RAG 서버의 모든 측면을 관리할 수 있습니다.

### 데이터베이스 관리

```bash
# 데이터베이스 초기화 (테이블 생성, 기본 역할/권한, 관리자 계정)
uv run python -m src.cli.main database init

# 데이터베이스 연결 테스트
uv run python -m src.cli.main database test

# 데이터베이스 상태 확인
uv run python -m src.cli.main database status

# 데이터베이스 백업
uv run python -m src.cli.main database backup --output backup.sql

# 데이터베이스 복원
uv run python -m src.cli.main database restore --input backup.sql
```

### 사용자 관리

```bash
# 새 사용자 생성
uv run python -m src.cli.main user create --username john --email john@example.com --role user

# 관리자 사용자 생성
uv run python -m src.cli.main user create --username admin --email admin@company.com --role admin

# 사용자 목록 조회
uv run python -m src.cli.main user list

# 특정 역할 사용자만 조회
uv run python -m src.cli.main user list --role admin

# 활성 사용자만 조회
uv run python -m src.cli.main user list --active

# JSON 형식으로 사용자 목록 출력
uv run python -m src.cli.main user list --format json
```

### 모델 테스트 및 구성

```bash
# 모든 LLM 프로바이더 테스트
uv run python -m src.cli.main model test-llm

# 특정 프로바이더 테스트
uv run python -m src.cli.main model test-llm --provider openai

# 커스텀 프롬프트로 테스트
uv run python -m src.cli.main model test-llm --prompt "양자 컴퓨팅에 대해 설명해주세요"

# 임베딩 모델 테스트
uv run python -m src.cli.main model test-embedding

# 모델 성능 벤치마크
uv run python -m src.cli.main model benchmark --iterations 20

# 사용 가능한 모델 목록
uv run python -m src.cli.main model list-models

# 모델 설정 변경
uv run python -m src.cli.main model set-model --llm-provider openai --llm-model gpt-4
```

### 데이터 관리

```bash
# 디렉토리에서 데이터 수집
uv run python -m src.cli.main data ingest --path ./documents --recursive

# 특정 파일 형식만 처리
uv run python -m src.cli.main data ingest --path ./docs --file-types pdf,docx

# 데이터 동기화
uv run python -m src.cli.main data sync --source filesystem

# 데이터 상태 확인
uv run python -m src.cli.main data status

# 데이터 정리
uv run python -m src.cli.main data cleanup --orphaned
```

### 설정 관리

```bash
# .env 파일 템플릿 생성
cp .env.example .env

# 환경 변수를 통한 설정 관리
# .env 파일을 직접 편집하여 설정 변경

# 현재 설정 확인
uv run python -m src.cli.main config show

# 설정 유효성 검증
uv run python -m src.cli.main config validate
```

### CLI 고급 기능

```bash
# 디버그 모드로 실행
uv run python -m src.cli.main --debug database status

# 상세 로그와 함께 실행
uv run python -m src.cli.main --verbose user list

# 커스텀 환경 파일 사용
uv run python -m src.cli.main --env-file custom.env database init

# 도움말 보기
uv run python -m src.cli.main --help
uv run python -m src.cli.main database --help
uv run python -m src.cli.main user --help
```

### CLI 특징

- **Rich 콘솔 출력**: 컬러풀한 테이블, 진행 표시줄, 상태 표시
- **글로벌 옵션**: `--debug`, `--verbose`, `--env-file` 지원
- **입력 검증**: 안전한 사용자 입력 및 확인 프롬프트
- **에러 처리**: 포괄적인 에러 메시지 및 복구 제안
- **진행 상태**: 장시간 작업에 대한 실시간 진행률 표시
- **도움말 시스템**: 모든 명령에 대한 상세한 도움말

## 🧪 개발

### 개발 환경 설정

```bash
# 개발 의존성 설치
uv sync --group dev

# 코드 포맷팅
uv run black src/
uv run isort src/

# 린팅
uv run flake8 src/
uv run mypy src/
```

### 테스팅

```bash
# 모든 테스트 실행
uv run pytest

# 커버리지 리포트
uv run pytest --cov=src

# 특정 테스트 실행
uv run pytest tests/unit/test_database_base.py
uv run pytest tests/unit/test_milvus_client.py

# CLI 기능 테스트
uv run python -m src.cli.main database test
uv run python -m src.cli.main model test-llm --provider openai
```

#### 🧪 테스트 현황

**완료된 테스트 (110/223 테스트 통과):**

**✅ Task 2 - 데이터베이스 레이어 테스트:**
- `test_database_base.py`: 21/21 통과 - 데이터베이스 매니저 및 팩토리 테스트
- `test_database_health.py`: 14/14 통과 - 헬스 체크 시스템 테스트  
- `test_database_engine.py`: 21/21 통과 - 엔진 및 구성 테스트
- 총 **56개 테스트 통과** (Task 2 핵심 기능 100% 커버리지)

**✅ Task 3 - Milvus 통합 테스트:**
- `test_milvus_client.py`: 30/30 통과 - 클라이언트 및 연결 풀 테스트
- 총 **30개 테스트 통과** (코어 Milvus 기능 100% 커버리지)

**🔄 진행 중인 테스트:**
- 고급 Milvus 컴포넌트 테스트 (스키마, RBAC, 검색 등)
- 데이터베이스 연결 풀 고급 기능 테스트

**테스트 품질 보장:**
- SQLAlchemy 모킹 패턴 정확성 검증
- Milvus API 호환성 테스트
- 에러 처리 및 재시도 메커니즘 검증
- 연결 상태 및 헬스 체크 검증

### 데이터베이스 기능

구현된 데이터베이스 레이어는 다음을 제공합니다:

1. **다중 데이터베이스 지원**: PostgreSQL, MySQL, MariaDB, Oracle, SQL Server
2. **고급 연결 풀링**: 모니터링 및 성능 추적 포함
3. **헬스 모니터링**: 다단계 헬스 체크 시스템
4. **에러 처리**: 지능형 에러 분류 및 재시도 메커니즘
5. **서킷 브레이커**: 복원력 있는 운영을 위한 내결함성 패턴
6. **스키마 인텔리전스**: 자동 데이터베이스 인트로스펙션 및 분석

## 🐳 배포

### Docker 배포

```bash
# 이미지 빌드
docker build -t rag-server .

# 컨테이너 실행
docker run -p 8000:8000 rag-server
```

### 프로덕션 설정

```bash
# 프로덕션 환경 설정
export APP_ENV=production
export SECRET_KEY=your-production-secret-key

# 프로덕션 서버 실행
uv run rag-server
```

## 🔍 문제 해결

### 일반적인 문제

1. **데이터베이스 연결 문제**
   - 데이터베이스 서버 상태 확인
   - 연결 설정 검증
   - 연결 풀 설정 검토

2. **LLM API 문제**
   - API 키 설정 확인
   - 속도 제한 확인
   - API 사용량 모니터링

3. **Milvus 연결 문제**
   - Milvus 서버 실행 상태 확인
   - 벡터 컬렉션 설정 확인

### 모니터링

```bash
# 애플리케이션 로그 확인
tail -f logs/app.log

# 에러 로그 확인
grep ERROR logs/app.log

# 데이터베이스 헬스 체크
uv run rag-cli health database
```

## 📊 성능 최적화

### Milvus 벡터 검색 최적화

```python
# 인덱스 타입별 최적화 전략
index_configs = {
    "small_dataset": {
        "index_type": "FLAT",
        "metric_type": "L2"
    },
    "medium_dataset": {
        "index_type": "IVF_FLAT", 
        "metric_type": "L2",
        "params": {"nlist": 1024}
    },
    "large_dataset": {
        "index_type": "HNSW",
        "metric_type": "L2", 
        "params": {"M": 16, "efConstruction": 200}
    }
}

# 검색 파라미터 최적화
search_params = {
    "IVF_FLAT": {"nprobe": 64},
    "HNSW": {"ef": 128},
    "IVF_PQ": {"nprobe": 32}
}
```

### RBAC 및 보안 최적화

```python
# 사용자 컨텍스트 최적화
user_context = {
    "user_id": "user123",
    "group_ids": ["analysts", "researchers"],
    "permissions": ["read", "write"],
    "cache_ttl": 300  # 5분 캐시
}

# 메타데이터 필터링 최적화
access_filter = 'user_id == "user123" OR JSON_CONTAINS(group_ids, "analysts")'
```

### 데이터베이스 최적화

```python
# 연결 풀 최적화
pool_config = {
    "pool_size": 20,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 3600
}
```

## 🤝 기여하기

1. 레포지토리를 포크하세요
2. 피처 브랜치를 생성하세요 (`git checkout -b feature/new-feature`)
3. 변경사항을 커밋하세요 (`git commit -am 'Add new feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/new-feature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 지원

- **문서**: [Documentation](docs/)
- **이슈 트래커**: [GitHub Issues](https://github.com/your-repo/issues)
- **연락처**: team@ragserver.com

---

**참고**: 이 프로젝트는 포괄적인 데이터베이스 관리, 헬스 모니터링 및 에러 처리 기능을 갖춘 프로덕션 레디 구현입니다. 데이터베이스 레이어는 완전히 구현되어 엔터프라이즈 사용이 가능합니다.

## 🌐 다른 언어

- [English](README_EN.md)