from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    """Model representing a single chat message"""
    timestamp: datetime
    sender: str
    content: str

    
class ChatAnalysisResult(BaseModel):
    """Results of chat analysis"""
    chat_id: str = Field(description="Unique identifier for this chat analysis")
    messages: List[ChatMessage]
    sentiment_analysis: Optional[Dict[str, Any]] = None
    language_patterns: Optional[Dict[str, Any]] = None
    relationship_metrics: Optional[Dict[str, Any]] = None 
    # segments: Optional[Dict[str, Any]] = None
    user_sentiment: Optional[Dict[str, Any]] = None
    vector_db_path: Optional[str] = Field(None, description="Path to the vector database for this chat")


class ChatSegmentSearchRequest(BaseModel):
    """Request model for searching chat segments"""
    user_id: str = Field(description="Unique identifier for the user")
    chat_id: str = Field(description="Unique identifier for the chat")
    query: str = Field(description="Search query text")
    top_k: int = Field(5, description="Maximum number of results to return")


class ChatSegmentSearchResult(BaseModel):
    """Result model for a single chat segment search match"""
    segment_id: str
    start_time: str
    end_time: str
    content: str
    similarity_score: float 