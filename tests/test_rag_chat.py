import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Remove or comment out the following line
# os.environ["TESTING"] = "0"


# Set testing environment variable
os.environ["TESTING"] = "1"

# Import modules after setting environment variable
from utils.rag_chat import rag_chat_manager
from utils.katalk_parser import parse_katalk_file
from utils.chat_segmenter import ChatSegmenter
from utils.vector_db import vector_db_manager

# Sample file paths for testing
KAKAOTALK_ANDROID_FILE = "KakaoTalkChats_android.txt"
IOS_CHAT_FILE = "Talk_2025.1.15 19_40-1_ios.txt"

@pytest.fixture
def chat_segmenter():
    """Return a ChatSegmenter instance for testing."""
    return ChatSegmenter()

@pytest.mark.asyncio
async def test_parse_chat_files():
    """Test that the chat parser can load both test files."""
    # Verify files exist
    assert os.path.exists(KAKAOTALK_ANDROID_FILE), f"Test file {KAKAOTALK_ANDROID_FILE} not found"
    assert os.path.exists(IOS_CHAT_FILE), f"Test file {IOS_CHAT_FILE} not found"
    
    # Parse Android file
    android_df = await parse_katalk_file(KAKAOTALK_ANDROID_FILE)
    assert not android_df.empty, "Android chat file parsing returned empty DataFrame"
    assert all(col in android_df.columns for col in ['datetime', 'sender', 'content']), "Android chat DataFrame missing required columns"
    
    # Parse iOS file
    ios_df = await parse_katalk_file(IOS_CHAT_FILE)
    assert not ios_df.empty, "iOS chat file parsing returned empty DataFrame"
    assert all(col in ios_df.columns for col in ['datetime', 'sender', 'content']), "iOS chat DataFrame missing required columns"
    
    # Log some basic stats
    print(f"Android chat: {len(android_df)} messages, {android_df['sender'].nunique()} participants")
    print(f"iOS chat: {len(ios_df)} messages, {ios_df['sender'].nunique()} participants")

@pytest.mark.asyncio
async def test_segment_chat(chat_segmenter):
    """Test that the chat segmenter can segment the test files."""
    # Parse files
    android_df = await parse_katalk_file(KAKAOTALK_ANDROID_FILE)
    ios_df = await parse_katalk_file(IOS_CHAT_FILE)
    
    # Add cleaned_content column (normally done by text_preprocessor)
    android_df['cleaned_content'] = android_df['content']
    ios_df['cleaned_content'] = ios_df['content']
    
    # Segment Android chat
    android_segments = await chat_segmenter.segment_chat(android_df)
    assert len(android_segments) > 0, "No segments found in Android chat"
    
    # Segment iOS chat
    ios_segments = await chat_segmenter.segment_chat(ios_df)
    assert len(ios_segments) > 0, "No segments found in iOS chat"
    
    # Log segment counts
    print(f"Android chat: {len(android_segments)} segments")
    print(f"iOS chat: {len(ios_segments)} segments")

@pytest.mark.asyncio
async def test_vector_db_embed_segments(chat_segmenter):
    """Test that the vector database can embed and store segments."""
    # Parse file and segment
    df = await parse_katalk_file(KAKAOTALK_ANDROID_FILE)
    df['cleaned_content'] = df['content']  # Simplified for testing
    segments = await chat_segmenter.segment_chat(df)
    
    # Test parameters
    user_id = "test_user"
    chat_id = "test_chat"
    
    # Embed and store segments
    db_path = await vector_db_manager.embed_and_store_segments(
        user_id=user_id,
        chat_id=chat_id,
        messages_df=df,
        segments=segments,
        chat_segmenter=chat_segmenter
    )
    
    assert db_path is not None, "Failed to create vector database"
    
    # Retrieve the vector database
    vector_db = await vector_db_manager.get_vector_db(user_id, chat_id)
    assert vector_db is not None, "Failed to retrieve vector database"

@pytest.mark.asyncio
async def test_chat_with_segments():
    """Test RAG chat functionality with embedded segments."""
    # Parse file, segment, and embed (reusing previous test steps)
    df = await parse_katalk_file(KAKAOTALK_ANDROID_FILE)
    df['cleaned_content'] = df['content']  # Simplified for testing
    
    chat_segmenter = ChatSegmenter()
    segments = await chat_segmenter.segment_chat(df)
    
    user_id = "test_user"
    chat_id = "test_chat"
    
    # Embed and store segments
    await vector_db_manager.embed_and_store_segments(
        user_id=user_id,
        chat_id=chat_id,
        messages_df=df,
        segments=segments,
        chat_segmenter=chat_segmenter
    )
    
    # Configure the mock chain
    mock_chain = MagicMock()
    mock_chain.return_value = {
        "answer": "This is a test response from the RAG system.",
        "source_documents": []
    }
    
    # # Testing the RAG chat with mocked components
    # with patch.object(rag_chat_manager, 'get_chat_chain', return_value=mock_chain):
    #     # Test the chat functionality
    response = await rag_chat_manager.chat_with_segments(
        user_id=user_id,
        chat_id=chat_id,
        message="생일이라던가 대화 참여자의 특별한 기념일이 언제인지 알려줘"  # "This is a test question. What was the restaurant name?"
    )
    print(response)
        # assert "response" in response, "Response missing from RAG chat output"
    assert len(response["response"]) > 0, "Empty response from RAG chat"

@pytest.mark.asyncio
async def test_chat_clear_history():
    """Test clearing chat history."""
    user_id = "test_user"
    chat_id = "test_chat"
    
    # Create a memory entry
    memory_key = f"{user_id}_{chat_id}"
    rag_chat_manager.conversation_memory[memory_key] = "test_memory"
    
    # Clear the history
    result = await rag_chat_manager.clear_chat_history(user_id, chat_id)
    assert result is True, "Failed to clear chat history"
    assert memory_key not in rag_chat_manager.conversation_memory, "Memory still exists after clearing"

if __name__ == "__main__":
    asyncio.run(test_parse_chat_files()) 