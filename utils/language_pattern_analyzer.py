import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import re
import logging
from collections import Counter
import asyncio
from kiwipiepy import Kiwi  # Changed from konlpy.tag import Okt
import datetime
import threading

logger = logging.getLogger(__name__)

# Thread-safe function for processing with Kiwi (no lock needed as Kiwi is thread-safe)
def process_with_kiwi(text, kiwi_instance):
    return kiwi_instance.analyze(text)

# Async batch processing function
async def process_texts(texts, kiwi_instance):
    loop = asyncio.get_running_loop()
    results = []
    
    # Process texts in parallel as Kiwi is thread-safe
    tasks = [loop.run_in_executor(None, lambda t=t: process_with_kiwi(t, kiwi_instance)) for t in texts]
    return await asyncio.gather(*tasks)

class LanguagePatternAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()  # Initialize Korean text processor with Kiwi
        self._cache = {}  # Simple cache for processed results
    
    async def analyze_language_patterns(
        self, 
        messages_df: pd.DataFrame,
        min_messages_per_user: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze language usage patterns from a conversation DataFrame.
        
        Args:
            messages_df: DataFrame containing conversation history with columns:
                         'sender', 'datetime', 'content', 'cleaned_content'
            min_messages_per_user: Minimum messages required for user analysis
            
        Returns:
            Dict containing language pattern metrics by user and overall stats
        """
        if messages_df.empty:
            logger.warning("Empty message dataframe provided for language pattern analysis")
            return {"error": "No messages to analyze"}
            
        # Create a cache key based on dataframe content hash
        cache_key = hash(str(messages_df.shape) + str(messages_df.iloc[0]['datetime']) + 
                         str(messages_df.iloc[-1]['datetime']))
         
        # Return cached results if available
        if cache_key in self._cache:
            logger.info("Returning cached language pattern analysis")
            return self._cache[cache_key]
        
        try:
            # Get unique users with sufficient message count
            user_message_counts = messages_df['sender'].value_counts()
            valid_users = user_message_counts[user_message_counts >= min_messages_per_user].index.tolist()
            
            if not valid_users:
                return {"error": f"No users with at least {min_messages_per_user} messages"}
            
            # Initialize results structure
            results = {
                "user_patterns": {},
                "overall_stats": {}
            }
            
            # Process each user's messages in parallel
            user_analysis_tasks = [
                self._analyze_user_patterns(messages_df[messages_df['sender'] == user], user)
                for user in valid_users
            ]
            
            user_results = await asyncio.gather(*user_analysis_tasks)
            
            # Combine user results
            for user, user_data in zip(valid_users, user_results):
                results["user_patterns"][user] = user_data
            
            # # Calculate comparative metrics between users
            # if len(valid_users) > 1:
            #     results["user_comparisons"] = await self._analyze_user_comparisons(
            #         results["user_patterns"]
            #     )
            
            # Cache the results
            self._cache[cache_key] = results
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing language patterns: {str(e)}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}
    
    async def _analyze_user_patterns(self, user_df: pd.DataFrame, username: str) -> Dict[str, Any]:
        """Analyze language patterns for a specific user"""
        if user_df.empty:
            return {}
            
        # Basic message statistics
        message_lengths = user_df['cleaned_content'].str.len()
        
        # Extract and tokenize Korean text
        all_text = " ".join(user_df['cleaned_content'].fillna(""))
        
        # Extract semantic keywords with POS tagging instead of simple tokenization
        semantic_keywords = await self._extract_semantic_keywords(all_text)
        
        # Message timing analysis
        if 'datetime' in user_df.columns:
            timing_analysis = await self._analyze_message_timing(user_df)
        else:
            timing_analysis = {}
        
        # Compile results
        return {
            "message_count": len(user_df),
            "message_length": {
                "mean": message_lengths.mean(),
                "median": message_lengths.median(),
                "std": message_lengths.std(),
                "max": message_lengths.max(),
                "min": message_lengths.min()
            },
            "vocabulary": {
                "semantic_keywords": semantic_keywords,
            },
            "timing": timing_analysis
        }
    
    async def _extract_semantic_keywords(self, text: str, top_k: int = 20) -> Dict[str, int]:
        """
        Extract semantically meaningful keywords from text.
        
        Args:
            text: The input text to analyze
            top_k: Number of top keywords to return
            
        Returns:
            Dictionary of top semantic keywords and their counts
        """
        # Run POS tagging using Kiwi (thread-safe)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.kiwi.tokenize(text))
        
        # Keep only meaningful parts of speech (nouns, verbs, adjectives)
        # For Korean with Kiwi: NNG/NNP (nouns), VV (verbs), VA (adjectives)
        semantic_pos = ['NNG', 'NNP', 'VV', 'VA']
        
        # Filter words by POS tags and count frequencies
        keyword_counter = Counter()
        
        # Kiwi의 analyze는 Token 객체 리스트를 반환합니다
        for token in results:
            word = token.form  # Token 객체의 form 속성에 단어가 있음
            pos = token.tag    # Token 객체의 tag 속성에 품사가 있음
            
            # Skip short words (likely less meaningful)
            if len(word) <= 1:
                continue
            
            # Keep only semantic parts of speech
            if pos in semantic_pos:
                keyword_counter[word] += 1
        
        # Return top k keywords with their counts
        return dict(keyword_counter.most_common(top_k))
    
    async def _tokenize_korean_text(self, text: str) -> List[str]:
        """Tokenize Korean text using Kiwipiepy for basic analysis"""
        # Kiwi is thread-safe so we can run it directly in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.kiwi.tokenize(text))
        
        # Token 객체에서 form(단어) 추출
        return [token.form for token in result]
    
    async def _analyze_message_timing(self, user_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when the user typically sends messages"""
        if 'datetime' not in user_df.columns or user_df.empty:
            return {}
            
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(user_df['datetime']):
            user_df['datetime'] = pd.to_datetime(user_df['datetime'])
            
        # Extract hour information
        user_df['hour'] = user_df['datetime'].dt.hour
        
        # Count messages by hour of day
        hour_counts = user_df['hour'].value_counts().sort_index()
        
        # Determine time of day preference - safely handle missing indices
        morning = hour_counts.loc[hour_counts.index.intersection(range(6, 12))].sum()
        afternoon = hour_counts.loc[hour_counts.index.intersection(range(12, 18))].sum()
        evening = hour_counts.loc[hour_counts.index.intersection(range(18, 24))].sum()
        night = hour_counts.loc[hour_counts.index.intersection([0, 1, 2, 3, 4, 5])].sum()
        
        total = morning + afternoon + evening + night
        
        # Return timing information
        return {
            "hour_distribution": hour_counts.to_dict(),
            "time_of_day_preference": {
                "morning": morning / total * 100 if total > 0 else 0,
                "afternoon": afternoon / total * 100 if total > 0 else 0,
                "evening": evening / total * 100 if total > 0 else 0,
                "night": night / total * 100 if total > 0 else 0
            }
        }
    
    async def _analyze_user_comparisons(self, user_patterns: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare language patterns between users"""
        if not user_patterns or len(user_patterns) < 2:
            return {}
            
        users = list(user_patterns.keys())
        comparisons = {}
        
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                user1 = users[i]
                user2 = users[j]
                
                user1_data = user_patterns[user1]
                user2_data = user_patterns[user2]
                
                # Skip if either user doesn't have complete data
                if not user1_data or not user2_data:
                    continue
                    
                # Calculate keyword similarity based on semantic keywords
                if 'vocabulary' in user1_data and 'vocabulary' in user2_data:
                    user1_keywords = set(user1_data['vocabulary'].get('semantic_keywords', {}).keys())
                    user2_keywords = set(user2_data['vocabulary'].get('semantic_keywords', {}).keys())
                    
                    common_keywords = user1_keywords.intersection(user2_keywords)
                    union_keywords = user1_keywords.union(user2_keywords)
                    
                    keyword_similarity = len(common_keywords) / len(union_keywords) if union_keywords else 0
                    
                    # Find distinctive keywords for each user
                    user1_distinctive = user1_keywords - user2_keywords
                    user2_distinctive = user2_keywords - user1_keywords
                else:
                    keyword_similarity = 0
                    user1_distinctive = set()
                    user2_distinctive = set()
                
                # Calculate messaging style differences
                style_diffs = {}
                
                # Compare message length
                if 'message_length' in user1_data and 'message_length' in user2_data:
                    ml1 = user1_data['message_length'].get('mean', 0)
                    ml2 = user2_data['message_length'].get('mean', 0)
                    style_diffs['message_length_diff'] = abs(ml1 - ml2)
                    style_diffs['message_length_ratio'] = max(ml1, ml2) / min(ml1, ml2) if min(ml1, ml2) > 0 else 0
                
                comparisons[f"{user1} vs {user2}"] = {
                    "keyword_similarity": keyword_similarity,
                    "distinctive_keywords": {
                        user1: list(user1_distinctive)[:10],  # Limit to top 10
                        user2: list(user2_distinctive)[:10]   # Limit to top 10
                    },
                    "style_differences": style_diffs
                }
        
        return comparisons

# Instantiate the analyzer as a singleton
language_pattern_analyzer = LanguagePatternAnalyzer() 