import os
import pytest
import asyncio
from fastapi import UploadFile
from pathlib import Path
from tempfile import NamedTemporaryFile

from services.chat_service import process_chat_file, search_chat_segments

# Test user ID for all tests
TEST_USER_ID = "test_user_123"

@pytest.fixture
def ios_chat_file():
    """Fixture for iOS chat file"""
    file_path = Path(__file__).parent.parent / "Talk_2025.1.15 19_40-1_ios.txt"
    
    # Ensure the file exists
    assert file_path.exists(), f"Test file not found: {file_path}"
    
    # Create an UploadFile object
    async def _get_upload_file():
        with open(file_path, "rb") as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="Talk_2025.1.15 19_40-1_ios.txt",
            file=NamedTemporaryFile(delete=False)
        )
        await upload_file.write(content)
        await upload_file.seek(0)
        return upload_file
    
    return _get_upload_file

@pytest.fixture
def android_chat_file():
    """Fixture for Android chat file"""
    file_path = Path(__file__).parent.parent / "KakaoTalkChats_android.txt"
    
    # Ensure the file exists
    assert file_path.exists(), f"Test file not found: {file_path}"
    
    # Create an UploadFile object
    async def _get_upload_file():
        with open(file_path, "rb") as f:
            content = f.read()
        
        upload_file = UploadFile(
            filename="KakaoTalkChats_android.txt",
            file=NamedTemporaryFile(delete=False)
        )
        await upload_file.write(content)
        await upload_file.seek(0)
        return upload_file
    
    return _get_upload_file

@pytest.mark.asyncio
async def test_process_ios_chat_file(ios_chat_file):
    """Test processing iOS chat file"""
    # Get the upload file
    upload_file = await ios_chat_file()
    
    try:
        # Process the chat file
        result = await process_chat_file(
            file=upload_file,
            user_id=TEST_USER_ID,
            options={}
        )
        
        # Verify the result
        assert result.chat_id is not None, "Chat ID should be assigned"
        assert len(result.messages) > 0, "Messages should be extracted"
        assert result.sentiment_analysis is not None, "Sentiment analysis should be performed"
        # assert result.segments is not None, "Chat segmentation should be performed"
        assert result.vector_db_path is not None, "Vector DB path should be returned"
        
        # Check first few messages
        first_message = result.messages[0]
        assert first_message.sender != "", "Message should have a sender"
        assert first_message.content != "", "Message should have content"
        assert first_message.timestamp is not None, "Message should have a timestamp"
        
        # Save the chat_id for segment search test
        return result.chat_id
    
    finally:
        # Cleanup
        await upload_file.close()
        if upload_file.file and hasattr(upload_file.file, "name"):
            try:
                # Force close file handles and add delay
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Try to delete with retry logic
                for i in range(5):
                    try:
                        if os.path.exists(upload_file.file.name):
                            os.unlink(upload_file.file.name)
                        break
                    except (OSError, PermissionError):
                        if i < 4:  # Don't sleep after last attempt
                            await asyncio.sleep(0.2 * (2 ** i))
                        else:
                            print(f"Warning: Could not delete temp file: {upload_file.file.name}")
            except:
                pass

@pytest.mark.asyncio
async def test_process_android_chat_file(android_chat_file):
    """Test processing Android chat file"""
    # Get the upload file
    upload_file = await android_chat_file()
    
    try:
        # Process the chat file
        result = await process_chat_file(
            file=upload_file,
            user_id=TEST_USER_ID,
            options={}
        )
        
        # Verify the result
        assert result.chat_id is not None, "Chat ID should be assigned"
        assert len(result.messages) > 0, "Messages should be extracted"
        assert result.sentiment_analysis is not None, "Sentiment analysis should be performed"
        # assert result.segments is not None, "Chat segmentation should be performed"
        assert result.vector_db_path is not None, "Vector DB path should be returned"
        
        # Check first few messages
        first_message = result.messages[0]
        assert first_message.sender != "", "Message should have a sender"
        assert first_message.content != "", "Message should have content"
        assert first_message.timestamp is not None, "Message should have a timestamp"
        
        # Save the chat_id for segment search test
        return result.chat_id
    
    finally:
        # Cleanup
        await upload_file.close()
        if upload_file.file and hasattr(upload_file.file, "name"):
            try:
                # Force close file handles and add delay
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Try to delete with retry logic
                for i in range(5):
                    try:
                        if os.path.exists(upload_file.file.name):
                            os.unlink(upload_file.file.name)
                        break
                    except (OSError, PermissionError):
                        if i < 4:  # Don't sleep after last attempt
                            await asyncio.sleep(0.2 * (2 ** i))
                        else:
                            print(f"Warning: Could not delete temp file: {upload_file.file.name}")
            except:
                pass

@pytest.mark.asyncio
async def test_search_chat_segments(ios_chat_file):
    """Test searching in chat segments after processing"""
    # First process the iOS chat file to populate the vector database
    upload_file = await ios_chat_file()
    
    try:
        # Process the chat file
        result = await process_chat_file(
            file=upload_file,
            user_id=TEST_USER_ID,
            options={}
        )
        
        chat_id = result.chat_id
        
        # Search for segments with various queries
        test_queries = ["밥 먹으러", "사진", "안녕하세요"]
        
        for query in test_queries:
            search_results = await search_chat_segments(
                user_id=TEST_USER_ID,
                chat_id=chat_id,
                query=query,
                top_k=3
            )
            
            # Verify search results
            assert isinstance(search_results, list), "Search results should be a list"
            # Note: some queries might not have matches, so we don't assert length > 0
            
            # If we have results, check their structure
            if search_results:
                first_result = search_results[0]
                assert "segment_id" in first_result, "Result should have segment_id"
                assert "content" in first_result, "Result should have content"
                assert "similarity_score" in first_result, "Result should have similarity_score"
                assert 0 <= first_result["similarity_score"] <= 1, "Similarity score should be between 0 and 1"
    
    finally:
        # Cleanup
        await upload_file.close()
        if upload_file.file and hasattr(upload_file.file, "name"):
            try:
                # Force close file handles and add delay
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Try to delete with retry logic
                for i in range(5):
                    try:
                        if os.path.exists(upload_file.file.name):
                            os.unlink(upload_file.file.name)
                        break
                    except (OSError, PermissionError):
                        if i < 4:  # Don't sleep after last attempt
                            await asyncio.sleep(0.2 * (2 ** i))
                        else:
                            print(f"Warning: Could not delete temp file: {upload_file.file.name}")
            except:
                pass

@pytest.mark.asyncio
async def test_end_to_end_chat_analysis(ios_chat_file, android_chat_file):
    """Test the full chat analysis flow with both iOS and Android files"""
    # Process iOS chat
    ios_upload_file = await ios_chat_file()
    android_upload_file = await android_chat_file()
    
    try:
        # Process iOS chat file
        ios_result = await process_chat_file(
            file=ios_upload_file,
            user_id=TEST_USER_ID,
            options={}
        )
        ios_chat_id = ios_result.chat_id
        
        # Process Android chat file
        android_result = await process_chat_file(
            file=android_upload_file,
            user_id=TEST_USER_ID,
            options={}
        )
        android_chat_id = android_result.chat_id
        
        assert ios_chat_id != android_chat_id, "Each chat should get a unique ID"
        
        # Try searching in both chats
        for chat_id in [ios_chat_id, android_chat_id]:
            # Search for a generic term that might exist in both chats
            search_results = await search_chat_segments(
                user_id=TEST_USER_ID,
                chat_id=chat_id,
                query="안녕",
                top_k=2
            )
            
            # We don't assert on results content because it depends on the actual chat data
            assert isinstance(search_results, list), "Search results should be a list"
    
    finally:
        # Cleanup
        await ios_upload_file.close()
        if ios_upload_file.file and hasattr(ios_upload_file.file, "name"):
            try:
                # Force close file handles and add delay
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Try to delete with retry logic
                for i in range(5):
                    try:
                        if os.path.exists(ios_upload_file.file.name):
                            os.unlink(ios_upload_file.file.name)
                        break
                    except (OSError, PermissionError):
                        if i < 4:  # Don't sleep after last attempt
                            await asyncio.sleep(0.2 * (2 ** i))
                        else:
                            print(f"Warning: Could not delete temp file: {ios_upload_file.file.name}")
            except:
                pass
                
        await android_upload_file.close()
        if android_upload_file.file and hasattr(android_upload_file.file, "name"):
            try:
                # Force close file handles and add delay
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Try to delete with retry logic
                for i in range(5):
                    try:
                        if os.path.exists(android_upload_file.file.name):
                            os.unlink(android_upload_file.file.name)
                        break
                    except (OSError, PermissionError):
                        if i < 4:  # Don't sleep after last attempt
                            await asyncio.sleep(0.2 * (2 ** i))
                        else:
                            print(f"Warning: Could not delete temp file: {android_upload_file.file.name}")
            except:
                pass 