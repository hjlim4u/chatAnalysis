import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError

from models.config import oauth_config
from models.auth import TokenData, OAuthProvider, OAuthUserInfo


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=oauth_config.jwt_expiration_minutes)
    
    to_encode.update({"exp": expire})
    
    # Create JWT token
    encoded_jwt = jwt.encode(
        to_encode, 
        oauth_config.jwt_secret_key, 
        algorithm=oauth_config.jwt_algorithm
    )
    
    return encoded_jwt


async def get_token_data(token: str = Depends(oauth2_scheme)) -> TokenData:
    """
    Validate and extract data from a JWT token.
    
    Args:
        token: JWT token to validate
        
    Returns:
        TokenData object containing the decoded token data
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token, 
            oauth_config.jwt_secret_key, 
            algorithms=[oauth_config.jwt_algorithm]
        )
        
        # Extract user ID
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Extract token expiration
        token_expiration = payload.get("exp")
        
        # Extract provider
        provider = payload.get("provider")
        if provider is None:
            raise credentials_exception
        
        # Create TokenData object
        token_data = TokenData(
            sub=user_id,
            exp=datetime.fromtimestamp(token_expiration) if token_expiration else None,
            provider=provider,
            scopes=payload.get("scopes", [])
        )
        
        return token_data
        
    except (JWTError, ValidationError):
        raise credentials_exception


async def get_oauth_client(
    provider: OAuthProvider, 
    redirect_uri: Optional[str] = None
) -> AsyncOAuth2Client:
    """
    Create an OAuth2 client for the specified provider.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        redirect_uri: Optional redirect URI
        
    Returns:
        AsyncOAuth2Client configured for the specified provider
    """
    client_kwargs = {"scope": oauth_config.oauth_endpoints[provider]["scope"]}
    
    if provider == OAuthProvider.GOOGLE:
        client = AsyncOAuth2Client(
            client_id=oauth_config.google_client_id,
            client_secret=oauth_config.google_client_secret,
            redirect_uri=redirect_uri or oauth_config.google_redirect_uri,
            **client_kwargs
        )
    elif provider == OAuthProvider.NAVER:
        client = AsyncOAuth2Client(
            client_id=oauth_config.naver_client_id,
            client_secret=oauth_config.naver_client_secret,
            redirect_uri=redirect_uri or oauth_config.naver_redirect_uri,
            **client_kwargs
        )
    elif provider == OAuthProvider.KAKAO:
        client = AsyncOAuth2Client(
            client_id=oauth_config.kakao_client_id,
            client_secret=oauth_config.kakao_client_secret,
            redirect_uri=redirect_uri or oauth_config.kakao_redirect_uri,
            **client_kwargs
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return client


async def get_oauth_login_url(provider: OAuthProvider, redirect_uri: Optional[str] = None) -> str:
    """
    Generate an OAuth login URL for the specified provider.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        redirect_uri: Optional redirect URI to override the default
        
    Returns:
        URL for OAuth login
    """
    client = await get_oauth_client(provider, redirect_uri)
    auth_url = oauth_config.oauth_endpoints[provider]["auth_url"]
    
    # Generate authorization URL
    uri, state = client.create_authorization_url(auth_url)
    
    return uri


async def get_oauth_token(
    provider: OAuthProvider, 
    code: str, 
    redirect_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exchange an authorization code for an OAuth token.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        code: Authorization code
        redirect_uri: Optional redirect URI
        
    Returns:
        Dictionary containing token information
        
    Raises:
        HTTPException: If token exchange fails
    """
    client = await get_oauth_client(provider, redirect_uri)
    token_url = oauth_config.oauth_endpoints[provider]["token"]
    
    try:
        # Exchange code for token
        token = await client.fetch_token(token_url, code=code)
        return token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve token: {str(e)}"
        )


async def get_oauth_user_info(
    provider: OAuthProvider, 
    token: Dict[str, Any]
) -> OAuthUserInfo:
    """
    Retrieve user information from the OAuth provider.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        token: OAuth token information
        
    Returns:
        OAuthUserInfo containing normalized user data
        
    Raises:
        HTTPException: If retrieving user info fails
    """
    userinfo_url = oauth_config.oauth_endpoints[provider]["userinfo_url"]
    access_token = token.get("access_token")
    
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token information"
        )
    
    try:
        # Fetch user info based on provider
        if provider == OAuthProvider.GOOGLE:
            return await _get_google_user_info(userinfo_url, access_token)
        elif provider == OAuthProvider.NAVER:
            return await _get_naver_user_info(userinfo_url, access_token)
        elif provider == OAuthProvider.KAKAO:
            return await _get_kakao_user_info(userinfo_url, access_token)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve user information: {str(e)}"
        )


async def _get_google_user_info(url: str, access_token: str) -> OAuthUserInfo:
    """
    Process Google-specific user information.
    
    Args:
        url: Google userinfo URL
        access_token: OAuth access token
        
    Returns:
        Normalized OAuthUserInfo
    """
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        
        user_data = response.json()
        
        return OAuthUserInfo(
            provider=OAuthProvider.GOOGLE,
            provider_user_id=user_data.get("sub"),
            email=user_data.get("email"),
            name=user_data.get("name"),
            profile_image=user_data.get("picture"),
            raw_user_info=user_data
        )


async def _get_naver_user_info(url: str, access_token: str) -> OAuthUserInfo:
    """
    Process Naver-specific user information.
    
    Args:
        url: Naver userinfo URL
        access_token: OAuth access token
        
    Returns:
        Normalized OAuthUserInfo
    """
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        if response_data.get("resultcode") != "00":
            raise ValueError(f"Error from Naver API: {response_data.get('message')}")
        
        user_data = response_data.get("response", {})
        
        return OAuthUserInfo(
            provider=OAuthProvider.NAVER,
            provider_user_id=user_data.get("id"),
            email=user_data.get("email"),
            name=user_data.get("name"),
            profile_image=user_data.get("profile_image"),
            raw_user_info=user_data
        )


async def _get_kakao_user_info(url: str, access_token: str) -> OAuthUserInfo:
    """
    Process Kakao-specific user information.
    
    Args:
        url: Kakao userinfo URL
        access_token: OAuth access token
        
    Returns:
        Normalized OAuthUserInfo
    """
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        
        user_data = response.json()
        account_data = user_data.get("kakao_account", {})
        profile_data = account_data.get("profile", {})
        
        return OAuthUserInfo(
            provider=OAuthProvider.KAKAO,
            provider_user_id=str(user_data.get("id")),
            email=account_data.get("email"),
            name=profile_data.get("nickname"),
            profile_image=profile_data.get("profile_image_url"),
            raw_user_info=user_data
        ) 