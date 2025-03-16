from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from enum import Enum


class OAuthProvider(str, Enum):
    """Enum for supported OAuth providers."""
    GOOGLE = "google"
    NAVER = "naver"
    KAKAO = "kakao"


class TokenData(BaseModel):
    """Schema for the data contained in authentication tokens."""
    sub: str
    exp: Optional[datetime] = None
    provider: OAuthProvider
    scopes: List[str] = []


class Token(BaseModel):
    """Schema for authentication tokens."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    name: str
    provider: OAuthProvider
    provider_user_id: str
    profile_image: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user information in responses."""
    id: str
    email: EmailStr
    name: str
    provider: OAuthProvider
    profile_image: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class OAuthUserInfo(BaseModel):
    """Schema for storing user information from OAuth providers."""
    provider: OAuthProvider
    provider_user_id: str
    email: EmailStr
    name: str
    profile_image: Optional[str] = None
    raw_user_info: Dict[str, Any] = Field(default_factory=dict)


class OAuthRequest(BaseModel):
    """Schema for requesting OAuth authentication."""
    code: str
    redirect_uri: str 