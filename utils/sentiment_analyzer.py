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
    Sentiment analyzer for chat messages using a pre-trained Korean sentiment analysis model.
    Implements singleton pattern to ensure model is loaded only once.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "Copycats/koelectra-base-v3-generalized-sentiment-analysis"):
        """
        Initialize the sentiment analyzer with the specified model.
        
        Args:
            model_name: The name of the HuggingFace model to use.
                        Default is Copycats/koelectra-base-v3-generalized-sentiment-analysis which is
                        a generalized Korean sentiment analysis model trained on multiple domains.
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
            return {"positive": 0.0, "negative": 1.0}
            
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
            
            # For Copycats/koelectra-base-v3-generalized-sentiment-analysis model:
            # Label 0 = negative, Label 1 = positive
            if self.num_labels == 2:
                sentiment_scores = {
                    "negative": float(probs[0]),
                    "positive": float(probs[1])
                }
            else:
                # Fallback for other models with more than 2 classes
                # Convert to binary classification by mapping
                sentiment_scores = {
                    "negative": 0.0,
                    "positive": 0.0
                }
                
                # Map based on label names in id2label if available
                if self.id2label:
                    for idx, label in self.id2label.items():
                        idx = int(idx)
                        label_lower = label.lower()
                        if "긍정" in label_lower or "positive" in label_lower or label == "1":
                            sentiment_scores["positive"] = float(probs[idx])
                        elif "부정" in label_lower or "negative" in label_lower or label == "0":
                            sentiment_scores["negative"] = float(probs[idx])
                else:
                    # Generic mapping for multi-class: merge all non-positive as negative
                    if self.num_labels > 2:
                        for i, prob in enumerate(probs):
                            # Last class is usually positive in multi-class sentiment
                            if i == self.num_labels - 1:
                                sentiment_scores["positive"] = float(prob)
                            else:
                                sentiment_scores["negative"] += float(prob)
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return {"positive": 0.0, "negative": 1.0}
    
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
        df['negative'] = df['sentiment'].apply(lambda x: x.get('negative', 0))
        
        # Calculate dominant sentiment (only positive or negative)
        df['dominant_sentiment'] = df.apply(
            lambda row: 'positive' if row['positive'] >= row['negative'] else 'negative', 
            axis=1
        )
        
        # Analyze sentiment by user
        user_sentiment = {}
        for user, user_df in df.groupby('sender'):
            user_sentiment[user] = {
                'average_sentiment': {
                    'positive': user_df['positive'].mean(),
                    'negative': user_df['negative'].mean()
                },
                'dominant_sentiment': user_df['dominant_sentiment'].mode()[0],
                'message_count': len(user_df),
                'sentiment_distribution': {
                    'positive': sum(user_df['dominant_sentiment'] == 'positive'),
                    'negative': sum(user_df['dominant_sentiment'] == 'negative')
                },
                'sentiment_ratio': {
                    'positive_ratio': sum(user_df['dominant_sentiment'] == 'positive') / len(user_df),
                    'negative_ratio': sum(user_df['dominant_sentiment'] == 'negative') / len(user_df)
                }
            }
        
        # Calculate overall sentiment
        overall_sentiment = {
            'average_sentiment': {
                'positive': df['positive'].mean(),
                'negative': df['negative'].mean()
            },
            'dominant_sentiment': df['dominant_sentiment'].mode()[0],
            'message_count': len(df),
            'sentiment_distribution': {
                'positive': sum(df['dominant_sentiment'] == 'positive'),
                'negative': sum(df['dominant_sentiment'] == 'negative')
            },
            'sentiment_ratio': {
                'positive_ratio': sum(df['dominant_sentiment'] == 'positive') / len(df),
                'negative_ratio': sum(df['dominant_sentiment'] == 'negative') / len(df)
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
            'negative': 'mean', 
            'dominant_sentiment': lambda x: sum(x == 'positive') / len(x)  # Positive ratio
        }).reset_index()
        
        return {
            'dates': daily_sentiment['date'].astype(str).tolist(),
            'positive': daily_sentiment['positive'].tolist(),
            'negative': daily_sentiment['negative'].tolist(),
            'positive_ratio': daily_sentiment['dominant_sentiment'].tolist()
        }

# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer() 