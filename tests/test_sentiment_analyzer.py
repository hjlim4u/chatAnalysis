import os
import sys
import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the sentiment analyzer
from utils.sentiment_analyzer import SentimentAnalyzer, sentiment_analyzer

# Test data with a mix of positive, negative, and neutral Korean texts
TEST_TEXTS = {
    "positive": [
        "오늘 정말 행복한 하루였어요! 모든 일이 잘 풀렸어요.",
        "이 영화는 정말 재미있고 감동적이었어요.",
        "네가 준 선물 정말 마음에 들어! 고마워!"
    ],
    "negative": [
        "오늘 정말 최악이었어. 모든 게 다 잘못됐어.",
        "이 식당 음식은 정말 맛없고 서비스도 형편없었어요.",
        "너무 실망스럽고 화가 나네요."
    ],
    "neutral": [
        "오늘 날씨는 흐리고 기온은 20도입니다.",
        "내일 회의는 오후 2시에 시작합니다.",
        "이 책은 총 300페이지로 구성되어 있습니다."
    ]
}

class TestSentimentAnalyzer:
    """Test cases for the SentimentAnalyzer class"""
    
    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that SentimentAnalyzer implements singleton pattern correctly"""
        analyzer1 = SentimentAnalyzer()
        analyzer2 = SentimentAnalyzer()
        
        # Both instances should be the same object
        assert analyzer1 is analyzer2
        
        # Both should be the same as the pre-created singleton
        assert analyzer1 is sentiment_analyzer
    
    @pytest.mark.asyncio
    async def test_analyze_individual_texts(self):
        """Test sentiment analysis on individual example texts"""
        # Print the model configuration to understand label structure
        print(f"\nModel configuration:")
        print(f"Number of labels: {sentiment_analyzer.num_labels}")
        print(f"Label mapping: {sentiment_analyzer.id2label}")
        
        for sentiment_type, texts in TEST_TEXTS.items():
            for text in texts:
                # Test the internal _analyze_text method directly
                result = sentiment_analyzer._analyze_text(text)
                
                # Print results for inspection
                print(f"\nExpected sentiment: {sentiment_type}")
                print(f"Text: {text}")
                print(f"Result: {result}")
                
                # Basic validation - ensure sentiment scores are present
                assert 'positive' in result
                assert 'negative' in result
                
                # Check that all scores are between 0 and 1
                for score in result.values():
                    assert 0 <= score <= 1
                
                # Verify that the scores sum approximately to 1
                total = sum(result.values())
                assert 0.95 <= total <= 1.05, f"Expected scores to sum approximately to 1, got {total}"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_by_user(self):
        """Test sentiment analysis on a sample conversation DataFrame"""
        # Create a sample DataFrame with the required columns
        now = datetime.now()
        
        # Create test data with two users
        data = []
        users = ["User1", "User2"]
        
        # Add positive messages for User1
        for i, text in enumerate(TEST_TEXTS["positive"]):
            data.append({
                "sender": users[0],
                "cleaned_content": text,
                "content": text,
                "datetime": now + timedelta(minutes=i*5)
            })
        
        # Add negative messages for User2
        for i, text in enumerate(TEST_TEXTS["negative"]):
            data.append({
                "sender": users[1],
                "cleaned_content": text,
                "content": text,
                "datetime": now + timedelta(minutes=i*5 + 2)
            })
        
        # Add neutral messages alternating between users
        for i, text in enumerate(TEST_TEXTS["neutral"]):
            data.append({
                "sender": users[i % 2],
                "cleaned_content": text,
                "content": text,
                "datetime": now + timedelta(minutes=i*5 + 15)
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        print(f"\nCreated test DataFrame with {len(df)} messages")
        
        # Run sentiment analysis
        results = await sentiment_analyzer.analyze_sentiment_by_user(df)
        
        # Basic validation
        assert "overall_sentiment" in results
        assert "user_sentiment" in results
        assert "sentiment_over_time" in results
        
        # Check user-specific results
        for user in users:
            assert user in results["user_sentiment"]
            user_result = results["user_sentiment"][user]
            
            # Check structure of user results
            assert "average_sentiment" in user_result
            assert "dominant_sentiment" in user_result
            assert "message_count" in user_result
            assert "sentiment_distribution" in user_result
            
   
        
        # Print results for inspection
        print("\nOverall sentiment:", results["overall_sentiment"])
        print("\nUser sentiment:", results["user_sentiment"])
        print("\nSentiment over time:", results["sentiment_over_time"])
        
        # Verify User1 is more positive than User2
        user1_positive = results["user_sentiment"]["User1"]["average_sentiment"]["positive"]
        user2_positive = results["user_sentiment"]["User2"]["average_sentiment"]["positive"]
        assert user1_positive > user2_positive, f"Expected User1 to be more positive than User2. User1: {user1_positive}, User2: {user2_positive}"
        
        # Verify User2 is more negative than User1
        user1_negative = results["user_sentiment"]["User1"]["average_sentiment"]["negative"]
        user2_negative = results["user_sentiment"]["User2"]["average_sentiment"]["negative"]
        assert user2_negative > user1_negative, f"Expected User2 to be more negative than User1. User2: {user2_negative}, User1: {user1_negative}"
        
        # Verify sentiment_over_time structure
        if results["sentiment_over_time"]:
            assert "dates" in results["sentiment_over_time"]
            assert "positive" in results["sentiment_over_time"]
            assert "negative" in results["sentiment_over_time"]
    
    @pytest.mark.asyncio
    async def test_empty_dataframe(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame(columns=["sender", "cleaned_content", "datetime"])
        results = await sentiment_analyzer.analyze_sentiment_by_user(empty_df)
        
        # Should return empty results
        assert results["overall_sentiment"] == {}
        assert results["user_sentiment"] == {}

# Helper function to run a single test (useful for troubleshooting)
async def run_test():
    """Run a specific test for debugging purposes"""
    test = TestSentimentAnalyzer()
    
    print("Running test_sentiment_analyzer.py")
    print("===== Testing singleton pattern =====")
    await test.test_singleton_pattern()
    
    print("\n===== Testing individual text analysis =====")
    await test.test_analyze_individual_texts()
    
    print("\n===== Testing sentiment analysis by user =====")
    await test.test_analyze_sentiment_by_user()
    
    print("\n===== Testing empty DataFrame handling =====")
    await test.test_empty_dataframe()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    # Run asynchronously
    asyncio.run(run_test()) 