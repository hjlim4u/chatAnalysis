import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

class ChatSegmenter:
    """
    Chat segmentation utility that divides chat transcripts into logical segments
    based on time differences and semantic similarity.
    """
    
    def __init__(self, embedding_model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        """
        Initialize the chat segmenter with the specified embedding model.
        
        Args:
            embedding_model_name: Name of the pretrained model for Korean text embeddings
        """
        self.short_response_threshold = timedelta(minutes=30)  # 30 minutes threshold for same bin
        self.long_response_threshold = timedelta(hours=4)      # 4 hours threshold for different bins
        self.semantic_similarity_threshold = 0.5               # Threshold for semantic similarity
        
        # Initialize embedding model and tokenizer for Korean text
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        
        # Cache for computed embeddings to improve performance
        self._embedding_cache = {}
    
    async def segment_chat(self, messages_df: pd.DataFrame) -> List[List[datetime]]:
        """
        Segment chat messages based on time differences and semantic similarity.
        
        Args:
            messages_df: DataFrame containing conversation history with columns:
                         'sender', 'datetime', 'content', 'cleaned_content'
                         
        Returns:
            List of tuples containing segment information
        """
        if messages_df.empty:
            logger.warning("Empty message dataframe provided for segmentation")
            return []
        
        # Sort messages by datetime if not already sorted
        messages_df = messages_df.sort_values(by='datetime')
        
        # Calculate time differences between consecutive messages
        messages_df['time_diff'] = messages_df['datetime'].diff()
        
        # Initialize segments
        segments_range = []
        last_message = []
        last_sender = None
        
        # Process each message to determine segments
        for i, row in messages_df.iterrows():
                
            time_diff = row['time_diff']
            
            # Rule 2: If response time ≥ 4 hours, different bins
            if last_message == [] or time_diff >= self.long_response_threshold:
                segments_range.append([row['datetime'], row['datetime']])
                last_message = [row['cleaned_content']]
                last_sender = row['sender']
            # Rule 1: If response time ≤ 30 minutes, same bin
            elif last_message and time_diff <= self.short_response_threshold:
                                
                # If same sender as previous message, combine messages for semantic analysis
                if row['sender'] == last_sender:
                    last_message.append(row['cleaned_content'])
                else:
                    last_message = [row['cleaned_content']]
                    last_sender = row['sender']
                segments_range[-1][1] = row['datetime']
                    
                
            # Rule 3: For times between 30 mins and 4 hours, check semantic similarity
            else:
                current_message = [row['cleaned_content']]
                current_sender = row['sender']
                j = i+1
                while j < len(messages_df) and messages_df.iloc[j]['sender'] == current_sender and messages_df.iloc[j]['time_diff'] <= self.short_response_threshold:
                    current_message.append(messages_df.iloc[j]['cleaned_content'])
                    j += 1

                # Get embeddings for previous and current message
                if len(last_message) > 0:
                    # Helper function to flatten nested lists
                    def flatten_list(nested_list):
                        result = []
                        for item in nested_list:
                            if isinstance(item, list):
                                result.extend(flatten_list(item))
                            else:
                                result.append(item)
                        return result
                    
                    # Flatten the lists before joining
                    last_message_flattened = flatten_list(last_message)
                    current_message_flattened = flatten_list(current_message)
                    
                    similarity = await self._compute_semantic_similarity(
                        ' '.join(last_message_flattened), 
                        ' '.join(current_message_flattened)
                    )
                    
                    # If similar enough, add to current segment
                    if similarity >= self.semantic_similarity_threshold:
                        
                        # If same sender, combine messages for semantic analysis
                        if row['sender'] == last_sender:
                            # Extend instead of append to avoid nested lists
                            last_message.extend(current_message)
                        else:
                            last_message = current_message
                            last_sender = current_sender
                        segments_range[-1][1] = row['datetime']
                    else:
                        segments_range.append([row['datetime'], row['datetime']])
                        last_message = [row['cleaned_content']]
                        last_sender = current_sender
                else:
                    # If we have no messages to compare with, start a new segment
                    segments_range.append([row['datetime'], row['datetime']])
                    last_message = [row['cleaned_content']]
                    last_sender = row['sender']


        
        # Generate segment metrics
        # segment_metrics = self._calculate_segment_metrics(segments_range)
        
        return segments_range
    
    async def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two text strings using embeddings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Create cache keys
        cache_key1 = hash(text1)
        cache_key2 = hash(text2)
        
        # Get or compute embeddings for text1
        if cache_key1 in self._embedding_cache:
            embedding1 = self._embedding_cache[cache_key1]
        else:
            embedding1 = await self._get_text_embedding(text1)
            self._embedding_cache[cache_key1] = embedding1
        
        # Get or compute embeddings for text2
        if cache_key2 in self._embedding_cache:
            embedding2 = self._embedding_cache[cache_key2]
        else:
            embedding2 = await self._get_text_embedding(text2)
            self._embedding_cache[cache_key2] = embedding2
        
        # Compute cosine similarity
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        
        return cos_sim.item()
    
    async def _get_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Generate embeddings for a text string using the Korean language model.
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: Text embedding vector
        """
        if not text or len(text.strip()) == 0:
            return None
            
        # Tokenize the text and convert to tensor
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # Generate embeddings from the model
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use CLS token as sentence embedding
            embedding = outputs.last_hidden_state[0, 0, :]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    

    
