import os
import tempfile
import time
from typing import Dict, Any, List, Optional
from fastapi import UploadFile, HTTPException
import asyncio
import logging
import uuid
import gc
import json
import pandas as pd
from datetime import datetime

from utils.katalk_parser import parse_katalk_file
from utils.text_preprocessor import TextPreprocessor
from utils.sentiment_analyzer import sentiment_analyzer
from utils.chat_segmenter import ChatSegmenter
from utils.vector_db import vector_db_manager
from models.chat import ChatAnalysisResult, ChatMessage
from utils.language_pattern_analyzer import language_pattern_analyzer
# Get singleton instance or create new if needed
text_preprocessor = TextPreprocessor()
# Initialize the chat segmenter with Korean embedding model
chat_segmenter = ChatSegmenter(model_name="klue/roberta-base")
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types like Timestamps."""
    def default(self, obj):
        # Handle pandas Timestamp
        if pd.api.types.is_datetime64_any_dtype(type(obj)) or isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # Handle numpy types
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()
        # Handle other datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Let the base class handle anything else
        return super().default(obj)

def log_result_sample(result: Any, prefix: str = "Result", max_items: int = 10):
    """
    Log a sample of the result, handling different data types appropriately.
    
    Args:
        result: The result to log
        prefix: Prefix string for the log
        max_items: Maximum number of items to log
    """
    try:
        if isinstance(result, pd.DataFrame):
            sample = result.head(max_items)
            logger.info(f"{prefix} (first {min(max_items, len(result))} of {len(result)} rows):\n{sample}")
        elif isinstance(result, dict):
            sample = dict(list(result.items())[:max_items])
            logger.info(f"{prefix} (up to {max_items} items):\n{json.dumps(sample, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}")
        elif isinstance(result, list):
            sample = result[:max_items]
            logger.info(f"{prefix} (first {min(max_items, len(result))} of {len(result)} items):\n{json.dumps(sample, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)}")
        else:
            # For non-serializable objects, use str() representation
            try:
                json_str = json.dumps(result, ensure_ascii=False, cls=CustomJSONEncoder)
                logger.info(f"{prefix}: {json_str}")
            except:
                logger.info(f"{prefix}: {str(result)}")
    except Exception as e:
        logger.error(f"Error logging result: {str(e)}", exc_info=True)
        logger.info(f"{prefix}: [Unable to format result for logging]")

async def process_chat_file(
    file: UploadFile,
    user_id: str,
    options: Dict[str, Any] = None
) -> ChatAnalysisResult:
    """
    Process an uploaded chat file and return analysis results
    
    Args:
        file: The uploaded chat file
        user_id: Unique identifier for the user
        options: Optional processing parameters
        
    Returns:
        ChatAnalysisResult: Analysis results of the chat history
    """
    process_start_time = time.time()
    
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a .txt file."
        )
    
    # Generate a unique chat_id for this analysis
    chat_id = str(uuid.uuid4()) if options.get("chat_id") is None else options.get("chat_id")
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file_path = temp_file.name
        try:
            # Write uploaded file content to temp file
            content = await file.read()
            temp_file.write(content)
            
            # Parse the chat file - returns a DataFrame
            parse_start_time = time.time()
            messages_df = await parse_katalk_file(temp_file_path)
            parse_end_time = time.time()
            logger.info(f"Chat parsing completed in {parse_end_time - parse_start_time:.3f} seconds")
            log_result_sample(messages_df, "Parsed chat messages")
            
            # Preprocess the messages to remove irrelevant patterns
            # This adds a 'cleaned_content' column and filters out rows
            preprocess_start_time = time.time()
            # messages_df['cleaned_content'] = messages_df['content'].apply(text_preprocessor.clean_text)
            messages_df['cleaned_content'] = text_preprocessor.clean_text_vectorized(messages_df['content'])
            preprocess_end_time = time.time()
            logger.info(f"Text preprocessing completed in {preprocess_end_time - preprocess_start_time:.3f} seconds")
            log_result_sample(messages_df[['sender', 'content', 'cleaned_content']], "Preprocessed messages")
            
            # Perform sentiment analysis using the analyzer's method
            sentiment_start_time = time.time()
            sentiment_task = sentiment_analyzer.analyze_sentiment_by_user(messages_df)
            language_pattern_task = language_pattern_analyzer.analyze_language_patterns(messages_df)
            logger.info(f"Sentiment analysis task created in {time.time() - sentiment_start_time:.3f} seconds")

            # Process chat segments based on time differences and semantic similarity
            segment_start_time = time.time()
            segment_results = await chat_segmenter.segment_chat(messages_df)
            segment_end_time = time.time()
            logger.info(f"Chat segmentation completed in {segment_end_time - segment_start_time:.3f} seconds")
            log_result_sample(segment_results, "Chat segmentation results")
            
            # Embed segments and store in vector database
            embedding_start_time = time.time()
            db_path = await vector_db_manager.embed_and_store_segments(
                user_id=user_id,
                chat_id=chat_id,
                messages_df=messages_df,
                segments=segment_results,
                chat_segmenter=chat_segmenter
            )
            embedding_end_time = time.time()
            logger.info(f"Segment embedding and storage completed in {embedding_end_time - embedding_start_time:.3f} seconds")
            logger.info(f"Vector database path: {db_path}")
            
            # Wait for sentiment analysis to complete
            sentiment_wait_start_time = time.time()
            sentiment_results, language_pattern_results = await asyncio.gather(sentiment_task, language_pattern_task)
            sentiment_wait_end_time = time.time()
            logger.info(f"Sentiment analysis completed in {sentiment_wait_end_time - sentiment_wait_start_time:.3f} seconds")
            log_result_sample(sentiment_results, "Sentiment analysis results")
            log_result_sample(language_pattern_results, "Language pattern analysis results")
            
            # Convert the DataFrame to ChatMessage objects for the result
            conversion_start_time = time.time()
            messages = [
                ChatMessage(
                    timestamp=row['datetime'],
                    sender=row['sender'],
                    content=row['cleaned_content']
                )
                for _, row in messages_df.iterrows()
            ]
            conversion_end_time = time.time()
            logger.info(f"DataFrame to message conversion completed in {conversion_end_time - conversion_start_time:.3f} seconds")
            log_result_sample(messages, "Converted chat messages")
            
            # Create a result object with messages, segments, and sentiment analysis
            result = ChatAnalysisResult(
                chat_id=chat_id,
                messages=messages,
                sentiment_analysis=sentiment_results,
                # segments=segment_results,
                user_sentiment=sentiment_results.get("user_sentiment", {}),
                vector_db_path=db_path
            )
            
            process_end_time = time.time()
            logger.info(f"Total chat processing completed in {process_end_time - process_start_time:.3f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chat file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing chat file: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            # Close any open file handles first by forcing garbage collection
            gc.collect()
            
            # Add a small delay to ensure file handles are released
            await asyncio.sleep(0.1)
            
            # Try multiple times to delete the file with exponential backoff
            for i in range(5):
                try:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    break
                except PermissionError:
                    if i < 4:  # Don't sleep after the last attempt
                        await asyncio.sleep(0.2 * (2 ** i))  # Exponential backoff
                    else:
                        logger.warning(f"Could not delete temporary file: {temp_file_path}")

async def search_chat_segments(
    user_id: str,
    chat_id: str,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for chat segments that match the query.
    
    Args:
        user_id: Unique identifier for the user
        chat_id: Unique identifier for the chat
        query: The search query
        top_k: Maximum number of results to return
        
    Returns:
        List of matched segments with similarity scores
    """
    search_start_time = time.time()
    
    if not user_id or not chat_id:
        raise HTTPException(
            status_code=400,
            detail="User ID and Chat ID are required"
        )
        
    try:
        # Search in the vector database
        search_query_start_time = time.time()
        results = await vector_db_manager.search_similar_segments(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            top_k=top_k
        )
        search_query_end_time = time.time()
        logger.info(f"Vector database search completed in {search_query_end_time - search_query_start_time:.3f} seconds")
        log_result_sample(results, "Search results")
        
        search_end_time = time.time()
        logger.info(f"Total search operation completed in {search_end_time - search_start_time:.3f} seconds")
        
        return results
    except Exception as e:
        logger.error(f"Error searching chat segments: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error searching chat segments: {str(e)}"
        ) 