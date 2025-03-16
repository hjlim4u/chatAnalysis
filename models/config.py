from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, Optional


class OAuthConfig(BaseSettings):
    """OAuth configuration for supported providers."""
    google_client_id: Optional[str] = Field(None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(None, env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: Optional[str] = Field(None, env="GOOGLE_REDIRECT_URI")
    
    naver_client_id: Optional[str] = Field(None, env="NAVER_CLIENT_ID")
    naver_client_secret: Optional[str] = Field(None, env="NAVER_CLIENT_SECRET")
    naver_redirect_uri: Optional[str] = Field(None, env="NAVER_REDIRECT_URI")
    
    kakao_client_id: Optional[str] = Field(None, env="KAKAO_CLIENT_ID")
    kakao_client_secret: Optional[str] = Field(None, env="KAKAO_CLIENT_SECRET")
    kakao_redirect_uri: Optional[str] = Field(None, env="KAKAO_REDIRECT_URI")
    
    # JWT Settings
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(60 * 24, env="JWT_EXPIRATION_MINUTES")  # 24 hours
    
    # OAuth provider endpoints - these could be moved to provider-specific classes if needed
    oauth_endpoints: Dict = {
        "google": {
            "auth_url": "https://accounts.google.com/o/oauth2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
            "scope": "openid email profile"
        },
        "naver": {
            "auth_url": "https://nid.naver.com/oauth2.0/authorize",
            "token_url": "https://nid.naver.com/oauth2.0/token",
            "userinfo_url": "https://openapi.naver.com/v1/nid/me",
            "scope": "name email profile_image"
        },
        "kakao": {
            "auth_url": "https://kauth.kakao.com/oauth/authorize",
            "token_url": "https://kauth.kakao.com/oauth/token",
            "userinfo_url": "https://kapi.kakao.com/v2/user/me",
            "scope": "profile_nickname account_email profile_image"
        }
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance for global access
oauth_config = OAuthConfig() 