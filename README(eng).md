# Chat Analysis API

A FastAPI application for analyzing chat data, focusing on sentiment analysis and relationship insights for Korean language conversations.

## Features

- **Advanced Korean Text Processing**: Specialized text cleaning pipeline for KakaoTalk chat exports
- **Emotion & Sentiment Analysis**: Analyze emotions and sentiment in Korean casual conversations
- **Conversation Insights**: Extract trends and patterns in conversations over time
- **Conversation Segmentation**: Automatically divide chats into meaningful segments
- **RAG-Enhanced Chat**: Interact with your chat history using retrieval-augmented generation
- **Vector Search**: Search through your chat history with semantic understanding
- **Authentication**: Secure API with token-based authentication
- **Fast and Asynchronous**: Non-blocking API for handling large chat histories efficiently

## Key Components

### Sentiment Analysis

- **Primary Model**: [snunlp/KR-ELECTRA-discriminator](https://huggingface.co/snunlp/KR-ELECTRA-discriminator) - A Korean ELECTRA model fine-tuned for sentiment analysis
- **Multi-emotion detection**: Identifies multiple emotions in a single message
- **Sentiment categorization**: Maps emotions to positive/negative/neutral sentiment categories
- **Conversation-level analysis**: Overall sentiment trends and distribution across conversations
- **Asynchronous processing**: Efficient handling of large chat histories
- **Caching**: Performance optimization with LRU caching for repeated phrases

### Conversation Segmentation and RAG

- **Chat segmentation**: Divides conversations into meaningful segments based on time and content
- **Vector embedding**: Stores chat segments in a vector database for semantic search
- **RAG chat**: Ask questions about your conversations and get AI-generated responses

## API Endpoints

### Authentication

```
POST /api/auth/signup
POST /api/auth/token
```

### Chat Analysis

```
POST /api/chat/analyze
```

Full analysis of an uploaded chat file with both message data and various analyses.

### Chat Segments

```
POST /api/chat/segments
POST /api/chat/segment-metrics
POST /api/chat/search
```

Endpoints for retrieving and searching through conversation segments.

### Interactive Chat

```
POST /api/chat/chat
POST /api/chat/clear/{chat_id}
```

Endpoints for chatting with your conversation history using RAG.

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment tool (venv, conda, etc.)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/chat-analysis.git
   cd chat-analysis
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application
   ```bash
   uvicorn main:app --reload
   ```

6. Access the API documentation at `http://localhost:8000/docs`

## Implementation Details

### Text Processing Pipeline

1. Chat message parsing from KakaoTalk exports (both Android and iOS formats)
2. Text cleaning with regex-based pattern removal
3. Preprocessing for Korean text specifics

### Sentiment Analysis Pipeline

1. Text cleaning using the text preprocessor
2. Emotion detection using KR-ELECTRA discriminator model
3. Mapping emotions to sentiment polarities (positive/negative/neutral)
4. Parallel processing of messages for efficiency
5. Sentiment aggregation for conversation-level insights
6. Result caching for performance optimization

### RAG Chat Pipeline

1. Segmentation of conversations into meaningful chunks
2. Vector embedding of segments for semantic search
3. Retrieval of relevant segments when user asks a question
4. Generation of responses with TinyLlama-based LLM
5. Conversation memory for contextual responses

## Project Structure

```
/chat-analysis
  ├── main.py                # FastAPI app initialization
  ├── routers/
  │     ├── auth_routes.py   # Authentication endpoints
  │     └── chat_routes.py   # Chat analysis endpoints
  ├── models/
  │     ├── auth.py          # Auth data models
  │     └── chat.py          # Chat data models  
  ├── services/
  │     └── chat_service.py  # Business logic for chat processing
  ├── utils/
  │     ├── sentiment_analyzer.py  # Sentiment analysis utilities
  │     ├── chat_segmenter.py      # Conversation segmentation
  │     ├── rag_chat.py            # RAG-enhanced chat functionality
  │     └── auth_utils.py          # Authentication utilities
  ├── tests/                # Test cases
  ├── data/                 # Data storage
  └── static/               # Static assets
```

## Dependencies

Key dependencies include:
- FastAPI and Pydantic v2 for API framework
- Transformers and PyTorch for NLP models
- LangChain for RAG functionality
- Sentence-Transformers for embeddings
- Redis for caching (optional)
- Python-Jose for JWT authentication

## License

MIT 