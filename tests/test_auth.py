import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from main import app
from models.schemas import OAuthProvider


client = TestClient(app)


@pytest.fixture
def mock_oauth_config():
    """Fixture to mock OAuth configuration."""
    with patch("utils.auth_utils.oauth_config") as mock_config:
        mock_config.jwt_secret_key = "test_secret_key"
        mock_config.jwt_algorithm = "HS256"
        mock_config.jwt_expiration_minutes = 60
        mock_config.oauth_endpoints = {
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
        yield mock_config


@pytest.fixture
def mock_get_oauth_login_url():
    """Fixture to mock the get_oauth_login_url function."""
    with patch("routers.auth_routes.get_oauth_login_url") as mock_func:
        mock_func.return_value = "https://accounts.google.com/o/oauth2/auth?redirect_uri=..."
        yield mock_func


@pytest.fixture
def mock_get_oauth_token():
    """Fixture to mock the get_oauth_token function."""
    with patch("routers.auth_routes.get_oauth_token") as mock_func:
        mock_func.return_value = {
            "access_token": "mock_oauth_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "mock_oauth_refresh_token"
        }
        yield mock_func


@pytest.fixture
def mock_get_oauth_user_info():
    """Fixture to mock the get_oauth_user_info function."""
    with patch("routers.auth_routes.get_oauth_user_info") as mock_func:
        mock_func.return_value = MagicMock(
            provider=OAuthProvider.GOOGLE,
            provider_user_id="123456789",
            email="test@example.com",
            name="Test User",
            profile_image="https://example.com/profile.jpg",
            raw_user_info={}
        )
        yield mock_func


@pytest.fixture
def mock_create_access_token():
    """Fixture to mock the create_access_token function."""
    with patch("routers.auth_routes.create_access_token") as mock_func:
        mock_func.return_value = "mock_jwt_access_token"
        yield mock_func


@pytest.mark.asyncio
async def test_login_endpoint(mock_get_oauth_login_url):
    """Test the login endpoint."""
    response = client.get("/auth/login/google")
    
    assert response.status_code == 200
    assert response.json() == {"login_url": "https://accounts.google.com/o/oauth2/auth?redirect_uri=..."}
    mock_get_oauth_login_url.assert_called_once()


@pytest.mark.asyncio
async def test_token_endpoint(
    mock_get_oauth_token, 
    mock_get_oauth_user_info, 
    mock_create_access_token
):
    """Test the token endpoint."""
    response = client.post(
        "/auth/token/google",
        json={"code": "test_code", "redirect_uri": "http://localhost:8000/auth/callback/google"}
    )
    
    assert response.status_code == 200
    assert response.json() == {
        "access_token": "mock_jwt_access_token",
        "token_type": "bearer",
        "expires_in": 1440 * 60,
        "refresh_token": "mock_oauth_refresh_token"
    }
    
    mock_get_oauth_token.assert_called_once()
    mock_get_oauth_user_info.assert_called_once()
    mock_create_access_token.assert_called_once()


@pytest.mark.asyncio
async def test_callback_endpoint():
    """Test the callback endpoint."""
    response = client.get("/auth/callback/google?code=test_code&state=test_state")
    
    assert response.status_code == 307  # Temporary redirect status code
    assert response.headers["location"].startswith("/auth/success")


@pytest.mark.asyncio
async def test_callback_endpoint_with_error():
    """Test the callback endpoint with error."""
    response = client.get("/auth/callback/google?error=access_denied")
    
    assert response.status_code == 307  # Temporary redirect status code
    assert response.headers["location"].startswith("/auth/error") 