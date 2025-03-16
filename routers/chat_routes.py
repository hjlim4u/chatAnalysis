from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.chat_service import process_chat_file, search_chat_segments
from ..models.chat import ChatAnalysisResult, ChatSegmentSearchRequest, ChatSegmentSearchResult
from ..models.auth import User
from ..utils.auth_utils import get_current_user
from ..utils.rag_chat import rag_chat_manager

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

class SegmentQueryParams(BaseModel):
    """Parameters for segment queries"""
    min_messages: Optional[int] = 5
    include_content: Optional[bool] = False
    sort_by: Optional[str] = "start_time"

class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message"""
    chat_id: str = Field(description="Unique identifier for the chat")
    message: str = Field(description="User message")

class ChatResponse(BaseModel):
    """Response model for chat messages"""
    response: str = Field(description="AI response text")
    contexts: List[Dict[str, Any]] = Field(description="Retrieved context segments")
    
@router.post("/analyze", response_model=ChatAnalysisResult)
async def analyze_chat(
    file: UploadFile = File(...),
    options: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
) -> ChatAnalysisResult:
    """
    Analyze the uploaded chat file and return full analysis results.
    Chat segments are embedded and stored in a vector database for later retrieval.
    
    Args:
        file: The chat export file to analyze
        options: Optional analysis parameters
        current_user: Current authenticated user
        
    Returns:
        ChatAnalysisResult: Full chat analysis results with a unique chat_id
    """
    try:
        # Process the chat file with user isolation
        result = await process_chat_file(
            file=file, 
            user_id=current_user.id,
            options=options
        )
        return result
    except Exception as e:
        logger.error(f"Error in chat analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing chat analysis: {str(e)}"
        )

@router.post("/segments", response_model=Dict[str, Any])
async def get_chat_segments(
    file: UploadFile = File(...),
    min_messages: int = Query(5, description="Minimum number of messages required for a segment"),
    include_content: bool = Query(False, description="Whether to include message content in response"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze the uploaded chat file and return segment analysis.
    
    Args:
        file: The chat export file to analyze
        min_messages: Minimum number of messages required for a segment
        include_content: Whether to include message content in response
        current_user: Current authenticated user
        
    Returns:
        Dict: Segment analysis results
    """
    try:
        result = await process_chat_file(file)
        
        # Extract just the segments information
        segment_data = result.segments
        
        # Filter segments based on min_messages
        if segment_data and 'segments' in segment_data:
            filtered_segments = [
                segment for segment in segment_data['segments']
                if segment['message_count'] >= min_messages
            ]
            
            # Optionally remove content to reduce response size
            if not include_content:
                for segment in filtered_segments:
                    for message in segment['messages']:
                        if 'content' in message:
                            message['content'] = ''
            
            # Update segment data with filtered segments
            segment_data['segments'] = filtered_segments
            segment_data['segment_count'] = len(filtered_segments)
        
        return segment_data
    except Exception as e:
        logger.error(f"Error in segment analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing segment analysis: {str(e)}"
        )

@router.post("/segment-metrics", response_model=Dict[str, Any])
async def get_segment_metrics(
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Get metrics about the chat segments.
    
    Args:
        file: The chat export file to analyze
        
    Returns:
        Dict: Segment metrics
    """
    try:
        result = await process_chat_file(file)
        
        # Extract just the segment metrics
        if result.segments and 'metrics' in result.segments:
            return {'metrics': result.segments['metrics']}
        return {'metrics': {}}
    except Exception as e:
        logger.error(f"Error getting segment metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing segment metrics: {str(e)}"
        )

@router.post("/search", response_model=List[Dict[str, Any]])
async def search_segments(
    search_request: ChatSegmentSearchRequest,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Search for chat segments that match the query using the vector database.
    
    Args:
        search_request: Search parameters including chat_id and query
        current_user: Current authenticated user
        
    Returns:
        List of matching segments with similarity scores
    """
    # Verify that the user has access to this chat
    if search_request.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to access this chat"
        )
    
    try:
        results = await search_chat_segments(
            user_id=current_user.id,
            chat_id=search_request.chat_id,
            query=search_request.query,
            top_k=search_request.top_k
        )
        return results
    except Exception as e:
        logger.error(f"Error searching chat segments: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error searching chat segments: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_history(
    chat_request: ChatMessageRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Chat with the analysis results using RAG to retrieve relevant context.
    
    Args:
        chat_request: Chat message and chat ID
        current_user: Current authenticated user
        
    Returns:
        Chat response with retrieved contexts
    """
    try:
        response = await rag_chat_manager.chat_with_segments(
            user_id=current_user.id,
            chat_id=chat_request.chat_id,
            message=chat_request.message
        )
        return response
    except Exception as e:
        logger.error(f"Error in chat with history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in chat with history: {str(e)}"
        )

@router.post("/chat/clear/{chat_id}")
async def clear_chat_history(
    chat_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Clear the conversation history for a specific chat.
    
    Args:
        chat_id: Unique identifier for the chat
        current_user: Current authenticated user
        
    Returns:
        Status message
    """
    try:
        success = await rag_chat_manager.clear_chat_history(
            user_id=current_user.id,
            chat_id=chat_id
        )
        
        if success:
            return {"status": "success", "message": "Chat history cleared successfully"}
        else:
            return {"status": "warning", "message": "No chat history found to clear"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing chat history: {str(e)}"
        ) 