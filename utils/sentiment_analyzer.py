import logging
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analyzer for chat messages using KoELECTRA model.
    Implements singleton pattern to ensure model is loaded only once.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "snunlp/KR-ELECTRA-discriminator"):
        """
        Initialize the sentiment analyzer with the specified model.
        
        Args:
            model_name: The name of the HuggingFace model to use
        """
        if self._initialized:
            return
            
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info(f"Loading sentiment analysis model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Check model configuration to determine label mapping
            self.num_labels = self.model.config.num_labels
            self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else None
            logger.info(f"Model has {self.num_labels} labels: {self.id2label}")
            
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize sentiment analyzer: {str(e)}")
    
    @lru_cache(maxsize=1024)
    def _analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a single text using the loaded model.
        Results are cached to avoid redundant processing.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
            
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
            
            # Use the model's configuration to map labels correctly
            if self.id2label:
                # Use the model's own label mapping
                sentiment_scores = {
                    "negative": 0.0,
                    "neutral": 0.0,
                    "positive": 0.0
                }
                
                # Map based on label names in id2label
                for idx, label in self.id2label.items():
                    label_lower = label.lower()
                    if "긍정" in label_lower or "positive" in label_lower:
                        sentiment_scores["positive"] = float(probs[int(idx)])
                    elif "부정" in label_lower or "negative" in label_lower:
                        sentiment_scores["negative"] = float(probs[int(idx)])
                    else:
                        sentiment_scores["neutral"] = float(probs[int(idx)])
            else:
                # Fallback mapping if id2label is not available
                # Adapt this based on the model's documentation
                if self.num_labels == 2:
                    # Binary classification (e.g., negative/positive)
                    sentiment_scores = {
                        "negative": float(probs[0]),
                        "neutral": 0.0,
                        "positive": float(probs[1])
                    }
                else:
                    # 3-class classification
                    sentiment_scores = {
                        "negative": float(probs[0]),
                        "neutral": float(probs[1]),
                        "positive": float(probs[2])
                    }
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    
    async def analyze_sentiment_by_user(self, messages_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment for each user in the conversation.
        
        Args:
            messages_df: DataFrame containing the chat messages with columns:
                        'sender', 'cleaned_content', 'datetime'
                        
        Returns:
            Dict containing sentiment analysis results by user and overall
        """
        if messages_df.empty:
            return {"overall_sentiment": {}, "user_sentiment": {}}
        
        # Make a copy to avoid modifying the original
        df = messages_df.copy()
        
        if 'cleaned_content' not in df.columns:
            if 'content' in df.columns:
                df['text_to_analyze'] = df['content']
            else:
                raise ValueError("DataFrame must contain either 'cleaned_content' or 'content' column")
        else:
            df['text_to_analyze'] = df['cleaned_content']
        
        # Filter out empty messages
        df = df[df['text_to_analyze'].notna() & (df['text_to_analyze'].str.strip() != "")]
        
        if df.empty:
            return {"overall_sentiment": {}, "user_sentiment": {}}
        
        # Process messages in batches using asyncio to prevent blocking
        async def process_batch(batch):
            loop = asyncio.get_event_loop()
            results = []
            
            for text in batch:
                # Run sentiment analysis in executor to not block the event loop
                result = await loop.run_in_executor(None, self._analyze_text, text)
                results.append(result)
                
            return results
        
        # Split messages into batches for processing
        batch_size = 32
        batches = [df['text_to_analyze'].tolist()[i:i+batch_size] 
                   for i in range(0, len(df), batch_size)]
        
        # Process all batches
        all_results = []
        for batch in batches:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
        
        # Add sentiment results to DataFrame
        df['sentiment'] = all_results
        
        # Extract sentiment scores
        df['positive'] = df['sentiment'].apply(lambda x: x.get('positive', 0))
        df['neutral'] = df['sentiment'].apply(lambda x: x.get('neutral', 0))
        df['negative'] = df['sentiment'].apply(lambda x: x.get('negative', 0))
        
        # Calculate dominant sentiment
        df['dominant_sentiment'] = df[['positive', 'neutral', 'negative']].idxmax(axis=1)
        
        # Analyze sentiment by user
        user_sentiment = {}
        for user, user_df in df.groupby('sender'):
            user_sentiment[user] = {
                'average_sentiment': {
                    'positive': user_df['positive'].mean(),
                    'neutral': user_df['neutral'].mean(),
                    'negative': user_df['negative'].mean()
                },
                'dominant_sentiment': user_df['dominant_sentiment'].mode()[0],
                'message_count': len(user_df),
                'sentiment_distribution': {
                    'positive': sum(user_df['dominant_sentiment'] == 'positive'),
                    'neutral': sum(user_df['dominant_sentiment'] == 'neutral'),
                    'negative': sum(user_df['dominant_sentiment'] == 'negative')
                }
            }
        
        # Calculate overall sentiment
        overall_sentiment = {
            'average_sentiment': {
                'positive': df['positive'].mean(),
                'neutral': df['neutral'].mean(),
                'negative': df['negative'].mean()
            },
            'dominant_sentiment': df['dominant_sentiment'].mode()[0],
            'message_count': len(df),
            'sentiment_distribution': {
                'positive': sum(df['dominant_sentiment'] == 'positive'),
                'neutral': sum(df['dominant_sentiment'] == 'neutral'),
                'negative': sum(df['dominant_sentiment'] == 'negative')
            }
        }
        
        return {
            "overall_sentiment": overall_sentiment,
            "user_sentiment": user_sentiment,
            "sentiment_over_time": self._calculate_sentiment_over_time(df)
        }
    
    def _calculate_sentiment_over_time(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate sentiment trends over time.
        
        Args:
            df: DataFrame with sentiment scores and timestamp
            
        Returns:
            Dict with sentiment trends
        """
        if 'datetime' not in df.columns:
            return {}
            
        # Sort by timestamp
        df = df.sort_values('datetime')
        
        # Group by day and calculate average sentiment
        df['date'] = df['datetime'].dt.date
        daily_sentiment = df.groupby('date').agg({
            'positive': 'mean',
            'neutral': 'mean', 
            'negative': 'mean'
        }).reset_index()
        
        return {
            'dates': daily_sentiment['date'].astype(str).tolist(),
            'positive': daily_sentiment['positive'].tolist(),
            'neutral': daily_sentiment['neutral'].tolist(),
            'negative': daily_sentiment['negative'].tolist()
        }

# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer() 