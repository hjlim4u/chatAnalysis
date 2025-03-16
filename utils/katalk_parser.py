import re
from datetime import datetime, timedelta
import pandas as pd
from typing import List
from models.chat import ChatMessage

def determine_date_pattern(first_few_lines: list[str]) -> str:
    """Determine the date pattern used in the chat file"""
    date_patterns = [
        r'(\d{4})\. (\d{1,2})\. (\d{1,2})\. (오전|오후) (\d{1,2}):(\d{2})',
        r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2})'
    ]
    
    for line in first_few_lines:
        for pattern in date_patterns:
            if re.search(pattern, line):
                return pattern
    return date_patterns[0]  # 기본값으로 첫 번째 패턴 반환

def parse_datetime(date_str: str, pattern: str) -> datetime | None:
    """Parse KakaoTalk datetime string to datetime object"""
    match = re.match(pattern, date_str)
    if match:
        year, month, day, ampm, hour, minute = match.groups()
        hour = int(hour)
        if ampm == '오후' and hour != 12:
            hour += 12
        elif ampm == '오전' and hour == 12:
            hour = 0
        return datetime(int(year), int(month), int(day), hour, int(minute))
    return None

def parse_katalk_message(line: str, date_pattern: str) -> dict | None:
    """Parse a single line of KakaoTalk message"""
    message_patterns = [
        f'({date_pattern.replace("(", "(?:")}), (.+?) : (.+)'
    ]
    
    for pattern in message_patterns:
        match = re.match(pattern, line)
        if match:
            datetime_str, sender, message = match.groups()
            parsed_datetime = parse_datetime(datetime_str, date_pattern)
            if parsed_datetime:
                return {
                    'datetime': parsed_datetime,
                    'sender': sender.strip(),
                    'content': message.strip()
                }
    return None

async def parse_katalk_file(file_path: str) -> pd.DataFrame:
    """
    Parse KakaoTalk chat export file to a pandas DataFrame
    
    Args:
        file_path: Path to the KakaoTalk export file
        
    Returns:
        pd.DataFrame: DataFrame with timestamp, sender, and content columns
    """
    messages = []
    current_message = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 첫 10줄 정도만 읽어서 날짜 패턴 파악
        first_lines = []
        for _ in range(10):
            line = f.readline().strip()
            if line:
                first_lines.append(line)
        
        date_pattern = determine_date_pattern(first_lines)
        
        # 파일 처음으로 되돌아가기
        f.seek(0)
        
        for line in f:
            line = line.strip()
            if not line or re.match(r'\d{4}년 \d{1,2}월 \d{1,2}일 \w요일', line):
                continue
                
            parsed = parse_katalk_message(line, date_pattern)
            if not parsed:
                continue
            
            if (not current_message) or (current_message[-1]['datetime'] == parsed['datetime']):
                current_message.append(parsed)
            else:
                # Distribute current_message evenly within the minute
                _distribute_messages_evenly(current_message, messages)
                current_message = [parsed]
                
    # Process any remaining messages
    for i, msg in enumerate(current_message):
        msg['datetime'] = msg['datetime'] + timedelta(seconds=i)
    messages.extend(current_message)
    
    # Convert messages to DataFrame more efficiently
    df = pd.DataFrame(messages)
    
    return df

def _distribute_messages_evenly(current_message: list, messages: list) -> None:
    """
    Distribute messages with the same timestamp evenly across a minute
    
    Args:
        current_message: List of messages with same timestamp
        messages: Main list to append distributed messages to
    """
    if not current_message:
        return
        
    # Number of messages in the batch
    msg_count = len(current_message)
    
    # Calculate interval between messages: 60 / (length + 1)
    # This creates spaces at the beginning and end of the minute
    interval = 60 / (msg_count + 1)
    
    # Distribute messages evenly
    for i, msg in enumerate(current_message):
        # Calculate seconds: interval * (i+1)
        # This gives positions at interval, 2*interval, 3*interval, etc.
        seconds = int(interval * (i + 1))
        msg['datetime'] = msg['datetime'].replace(second=seconds)
        messages.append(msg)