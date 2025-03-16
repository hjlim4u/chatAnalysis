import logging
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import torch
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd

logger = logging.getLogger(__name__)

class VectorDBManager:
    """
    Manages vector embeddings and retrieval for chat segments
    with proper isolation between users and chat rooms.
    """
    
    def __init__(self, embedding_model_name: str = "klue/roberta-base"):
        """
        Initialize the vector database manager with the specified embedding model.
        
        Args:
            embedding_model_name: Name of the pretrained model for embeddings
        """
        self.embedding_model_name = embedding_model_name
        logger.info(f"Initializing vector database with model: {embedding_model_name}")
        
        # Initialize HuggingFace embeddings for vector database
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Directory to store FAISS indices
        self.db_dir = os.path.join(os.getcwd(), "data", "vector_db")
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Cache of loaded vector databases by user_id and chat_id
        self.db_cache = {}
        
    async def embed_and_store_segments(
        self,
        user_id: str,
        chat_id: str,
        messages_df: pd.DataFrame,
        segments: List[List[datetime]],
        chat_segmenter
    ) -> str:
        """
        Embed chat segments and store them in a vector database,
        maintaining isolation between users and chat rooms.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat room
            messages_df: DataFrame containing the chat messages
            segments: List of segment start/end time tuples
            chat_segmenter: Instance of ChatSegmenter to extract segment content
            
        Returns:
            str: Path to the stored vector database
        """
        if not segments:
            logger.warning(f"No segments provided for user {user_id}, chat {chat_id}")
            return None
            
        # Create user and chat specific directory
        user_chat_dir = os.path.join(self.db_dir, user_id, chat_id)
        os.makedirs(user_chat_dir, exist_ok=True)
        
        # Extract text content for each segment
        segment_texts = []
        metadata_list = []
        
        for i, segment in enumerate(segments):
            # Get formatted text content for this segment
            segment_text = await self.get_segment_content(messages_df, segment)
            if not segment_text:
                continue
                
            # Store with metadata
            start_time, end_time = segment
            metadata = {
                "segment_id": f"seg_{i}_{start_time.strftime('%Y%m%d%H%M%S')}",
                "user_id": user_id,
                "chat_id": chat_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }
            
            segment_texts.append(segment_text)
            metadata_list.append(metadata)
        
        if not segment_texts:
            logger.warning(f"No valid segment texts extracted for user {user_id}, chat {chat_id}")
            return None
            
        # Create vector store
        logger.info(f"Creating vector store with {len(segment_texts)} segments")
        vector_store = await asyncio.to_thread(
            FAISS.from_texts,
            segment_texts,
            self.embeddings,
            metadatas=metadata_list
        )
        
        # Save to disk
        db_path = os.path.join(user_chat_dir, f"faiss_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        await asyncio.to_thread(vector_store.save_local, db_path)
        
        # Cache the vector store
        cache_key = f"{user_id}_{chat_id}"
        self.db_cache[cache_key] = vector_store
        
        logger.info(f"Vector database saved for user {user_id}, chat {chat_id} at {db_path}")
        return db_path
    
    async def get_vector_db(self, user_id: str, chat_id: str) -> Optional[FAISS]:
        """
        Get the vector database for a specific user and chat room.
        Loads from cache if available, otherwise loads from disk.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat room
            
        Returns:
            FAISS: The vector database instance
        """
        cache_key = f"{user_id}_{chat_id}"
        
        # Return from cache if available
        if cache_key in self.db_cache:
            return self.db_cache[cache_key]
            
        # Look for the most recent DB on disk
        user_chat_dir = os.path.join(self.db_dir, user_id, chat_id)
        if not os.path.exists(user_chat_dir):
            logger.warning(f"No vector database found for user {user_id}, chat {chat_id}")
            return None
            
        # Find the latest DB folder
        db_folders = [d for d in os.listdir(user_chat_dir) if d.startswith("faiss_")]
        if not db_folders:
            logger.warning(f"No FAISS database folders found for user {user_id}, chat {chat_id}")
            return None
            
        latest_db = sorted(db_folders)[-1]
        db_path = os.path.join(user_chat_dir, latest_db)
        
        # Load the vector store
        try:
            vector_store = await asyncio.to_thread(
                FAISS.load_local,
                db_path, 
                self.embeddings
            )
            
            # Cache for future use
            self.db_cache[cache_key] = vector_store
            logger.info(f"Loaded vector database for user {user_id}, chat {chat_id} from {db_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}", exc_info=True)
            return None
    
    async def search_similar_segments(
        self,
        user_id: str,
        chat_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for segments similar to the query.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat room
            query: The query text to search for similar segments
            top_k: Maximum number of results to return
            
        Returns:
            List of dictionaries containing segment information and similarity scores
        """
        vector_store = await self.get_vector_db(user_id, chat_id)
        if not vector_store:
            logger.warning(f"No vector database available for user {user_id}, chat {chat_id}")
            return []
            
        # Search for similar documents
        try:
            results = await asyncio.to_thread(
                vector_store.similarity_search_with_score,
                query,
                k=top_k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # FAISS similarity_search_with_score는 거리(distance)를 반환하므로 유사도로 변환
                # 거리가 클수록 유사도가 낮음 - 0~1 범위로 정규화
                
                # 거리를 유사도로 변환 (1/(1+거리) 공식 사용)
                normalized_score = 1.0 / (1.0 + float(score))
                
                formatted_results.append({
                    "segment_id": doc.metadata.get("segment_id"),
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time"),
                    "content": doc.page_content,
                    "similarity_score": normalized_score
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}", exc_info=True)
            return []
    
    async def get_segment_content(self, messages_df: pd.DataFrame, 
                                  segment: Tuple[datetime, datetime]) -> str:
        """
        Extract and format the content of a specific chat segment for embedding,
        concatenating consecutive messages from the same sender.
        
        Args:
            messages_df: DataFrame containing conversation history
            segment: Tuple of (start_datetime, end_datetime) defining the segment
            
        Returns:
            Formatted string containing all messages in the segment with metadata
        """
        start_time, end_time = segment
        
        # Filter messages within this segment's time range
        segment_messages = messages_df[
            (messages_df['datetime'] >= start_time) & 
            (messages_df['datetime'] <= end_time)
        ].sort_values('datetime')
        
        if segment_messages.empty:
            return ""
        
        # Build the segment content string with clear conversation markers
        segment_text = f"[SEGMENT_START] {start_time.isoformat()}\n"
        
        last_sender = None
        last_content = []
        
        for _, msg in segment_messages.iterrows():
            sender = msg['sender']
            content = msg['cleaned_content']
            
            # If this is a new sender, add the previous sender's combined messages
            if last_sender and sender != last_sender:
                segment_text += f"[{last_sender}] {' '.join(last_content)}\n"
                last_content = [content]
            # Otherwise, just add this message to the current batch
            else:
                last_content.append(content)
            
            last_sender = sender
        
        # Add the last sender's combined messages
        if last_sender and last_content:
            segment_text += f"[{last_sender}] {' '.join(last_content)}\n"
        
        segment_text += f"[SEGMENT_END] {end_time.isoformat()}"
        
        return segment_text

# Initialize a singleton instance
vector_db_manager = VectorDBManager() 