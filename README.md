# 채팅 분석 API

대화 데이터를 분석하여 감정 분석과 관계 통찰력을 제공하는 FastAPI 애플리케이션입니다. 특히 한국어 카카오톡 대화를 위해 최적화되어 있습니다.

## 주요 기능

- **고급 한국어 텍스트 처리**: 카카오톡 대화 내보내기 파일을 위한 특화된 텍스트 정제 파이프라인
- **감정 및 감성 분석**: 한국어 일상 대화에서 감정과 정서를 분석
- **대화 인사이트**: 시간에 따른 대화의 경향과 패턴 추출
- **대화 세그먼트화**: 대화를 의미 있는 단위로 자동 분할
- **RAG 기반 대화**: 검색 증강 생성(RAG)을 활용한 대화 기록과의 상호작용
- **벡터 검색**: 의미적 이해를 바탕으로 대화 기록 검색
- **인증**: 토큰 기반 보안 API
- **빠르고 비동기적**: 대규모 대화 기록을 효율적으로 처리하는 논블로킹 API

## 핵심 구성 요소

### 감성 분석

- **기본 모델**: [snunlp/KR-ELECTRA-discriminator](https://huggingface.co/snunlp/KR-ELECTRA-discriminator) - 감성 분석을 위해 미세 조정된 한국어 ELECTRA 모델
- **다중 감정 감지**: 단일 메시지에서 여러 감정 식별
- **감성 분류**: 감정을 긍정/부정/중립 범주로 매핑
- **대화 수준 분석**: 전체 대화의 감성 추세 및 분포
- **비동기 처리**: 대규모 대화 기록의 효율적 처리
- **캐싱**: 반복되는 문구에 대한 LRU 캐싱을 통한 성능 최적화

### 대화 세그먼트화 및 RAG

- **대화 분할**: 시간과 내용을 기반으로 대화를 의미 있는 세그먼트로 분할
- **벡터 임베딩**: 의미 검색을 위해 대화 세그먼트를 벡터 데이터베이스에 저장
- **RAG 채팅**: 대화에 대한 질문을 하고 AI 생성 응답 받기

## API 엔드포인트

### 인증

```
POST /api/auth/signup
POST /api/auth/token
```

### 채팅 분석

```
POST /api/chat/analyze
```

업로드된 채팅 파일에 대한 메시지 데이터와 다양한 분석을 포함한 전체 분석.

### 채팅 세그먼트

```
POST /api/chat/segments
POST /api/chat/segment-metrics
POST /api/chat/search
```

대화 세그먼트 검색 및 조회를 위한 엔드포인트.

### 대화형 채팅

```
POST /api/chat/chat
POST /api/chat/clear/{chat_id}
```

RAG를 사용하여 대화 기록과 채팅하기 위한 엔드포인트.

## 시작하기

### 사전 요구 사항

- Python 3.8 이상
- 가상 환경 도구 (venv, conda 등)

### 설치

1. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/chat-analysis.git
   cd chat-analysis
   ```

2. 가상 환경 생성 및 활성화
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

4. `.env.example`을 기반으로 `.env` 파일 생성
   ```bash
   cp .env.example .env
   # .env 파일을 원하는 설정으로 편집
   ```

5. 애플리케이션 실행
   ```bash
   uvicorn main:app --reload
   ```

6. `http://localhost:8000/docs`에서 API 문서 접근

## 구현 세부 사항

### 텍스트 처리 파이프라인

1. 카카오톡 내보내기 파일에서 채팅 메시지 파싱 (안드로이드 및 iOS 형식 모두 지원)
2. 정규식 기반 패턴 제거를 통한 텍스트 정제
3. 한국어 텍스트 특성에 맞는 전처리

### 감성 분석 파이프라인

1. 텍스트 전처리기를 사용한 텍스트 정제
2. KR-ELECTRA 판별자 모델을 사용한 감정 감지
3. 감정을 감성 극성(긍정/부정/중립)으로 매핑
4. 효율성을 위한 메시지 병렬 처리
5. 대화 수준 인사이트를 위한 감성 집계
6. 성능 최적화를 위한 결과 캐싱

### RAG 채팅 파이프라인

1. 대화를 의미 있는 청크로 세그먼트화
2. 의미 검색을 위한 세그먼트 벡터 임베딩
3. 사용자가 질문할 때 관련 세그먼트 검색
4. TinyLlama 기반 LLM을 통한 응답 생성
5. 맥락적 응답을 위한 대화 메모리

## 프로젝트 구조

```
/chat-analysis
  ├── main.py                # FastAPI 앱 초기화
  ├── routers/
  │     ├── auth_routes.py   # 인증 엔드포인트
  │     └── chat_routes.py   # 채팅 분석 엔드포인트
  ├── models/
  │     ├── auth.py          # 인증 데이터 모델
  │     └── chat.py          # 채팅 데이터 모델  
  ├── services/
  │     └── chat_service.py  # 채팅 처리 비즈니스 로직
  ├── utils/
  │     ├── sentiment_analyzer.py  # 감성 분석 유틸리티
  │     ├── chat_segmenter.py      # 대화 세그먼트화
  │     ├── rag_chat.py            # RAG 강화 채팅 기능
  │     └── auth_utils.py          # 인증 유틸리티
  ├── tests/                # 테스트 케이스
  ├── data/                 # 데이터 저장
  └── static/               # 정적 에셋
```

## 주요 의존성

주요 의존성 패키지:
- FastAPI와 Pydantic v2 (API 프레임워크)
- Transformers와 PyTorch (NLP 모델)
- LangChain (RAG 기능)
- Sentence-Transformers (임베딩)
- Redis (캐싱, 선택적)
- Python-Jose (JWT 인증)

## 라이센스

MIT