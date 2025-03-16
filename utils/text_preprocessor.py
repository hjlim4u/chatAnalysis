import re
from typing import List, Dict, Any, Optional, Pattern
import emoji
import pandas as pd
from pykospacing import Spacing


class TextPreprocessor:
    """
    Text preprocessing utility to clean chat messages by removing irrelevant patterns,
    emojis, and URLs before analysis. Also corrects spacing in Korean text.
    """
    
    def __init__(self):
        """Initialize pattern definitions for text cleaning"""
        # Korean character patterns and other cleaning patterns
        self.chat_patterns = {
            'single_consonants': r'[ㄱ-ㅎ]+',  # 자음만 있는 경우
            'single_vowels': r'[ㅏ-ㅣ]+',      # 모음만 있는 경우
            'media': r'^동영상$|^사진$|^사진 [0-9]{1,2}장$|^<(사진|동영상) 읽지 않음>$',
            # 필수적이지 않은 특수문자
            'special_chars': r'[~@#$%^&*()_+=`\[\]{}|\\<>]',
            # 시스템 메시지 패턴
            'system_messages': {
                'location': r'지도: .+',  # 위치 공유
                'map_share': r'\[네이버 지도\]',  # 지도 공유
                'audio_file': r'[a-f0-9]{64}\.m4a',  # 음성 메시지
                'music_share': r"'.+' 음악을 공유했습니다\.",  # 음악 공유
                'file_share': r'파일: .+\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|txt)$',  # 파일 공유
            }
        }
        
        # URL pattern
        self.url_pattern = r'https?://\S+|www\.\S+'
        
        # Initialize Korean spacing corrector
        self.spacing = Spacing()
        
        # Precompile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Precompile all regex patterns for improved performance"""
        # Compile main patterns
        self.compiled_patterns = {
            key: re.compile(pattern) for key, pattern in self.chat_patterns.items() 
            if key != 'system_messages'
        }
        
        # Compile system message patterns
        self.compiled_system_patterns = {
            key: re.compile(pattern) for key, pattern in self.chat_patterns['system_messages'].items()
        }
        
        # Compile URL pattern
        self.compiled_url_pattern = re.compile(self.url_pattern)
    
    def clean_text(self, text: str) -> str:
        """
        Remove irrelevant patterns, emojis, and URLs from text, and correct Korean spacing
        
        Args:
            text: The input text to clean
            
        Returns:
            str: Cleaned text with irrelevant patterns removed and spacing corrected
        """
        if not text:
            return ""
        
        # Remove URLs
        text = self.compiled_url_pattern.sub('', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove single consonants and vowels
        text = self.compiled_patterns['single_consonants'].sub('', text)
        text = self.compiled_patterns['single_vowels'].sub('', text)
        
        # Remove special characters
        text = self.compiled_patterns['special_chars'].sub('', text)
        
        # Check if the text matches any media pattern
        if self.compiled_patterns['media'].fullmatch(text):
            return ""
        
        # Check if the text matches any system message pattern
        for pattern in self.compiled_system_patterns.values():
            if pattern.search(text):
                return ""
        
        # Remove extra whitespace and strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Correct Korean spacing if text is not empty
        if text and any(ord('가') <= ord(char) <= ord('힣') for char in text):
            # Apply spacing correction only if Korean characters are present
            text = self.spacing(text)
        
        return text
    
    def process_messages(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of chat messages by cleaning their content and correcting spacing
        
        Args:
            messages_df: DataFrame with chat messages
            
        Returns:
            pd.DataFrame: DataFrame with original and cleaned content
        """
        # Add a cleaned_content column to the original DataFrame
        messages_df['cleaned_content'] = messages_df['content'].apply(self.clean_text)
        
        return messages_df

    def clean_text_vectorized(self, series):
        """Vectorized text cleaning for a pandas Series"""
        # Create a copy to avoid modifying the original
        cleaned = series.copy()
        
        # Apply regex replacements vectorized
        for pattern_name, pattern in self.compiled_patterns.items():
            cleaned = cleaned.str.replace(pattern, ' ', regex=True)
        
        # Additional vectorized operations
        cleaned = cleaned.str.lower()
        cleaned = cleaned.str.strip()
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        
        return cleaned