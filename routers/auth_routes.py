from datetime import timedelta, datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import RedirectResponse

from models.config import oauth_config
from models.auth import (
    OAuthProvider, 
    Token, 
    OAuthRequest, 
    TokenData,
    OAuthUserInfo,
    UserResponse
)
from utils.auth_utils import (
    create_access_token, 
    get_oauth_login_url, 
    get_oauth_token, 
    get_oauth_user_info,
    get_token_data
)

# Initialize the router
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}},
)


@router.get("/login/{provider}")
async def login(
    provider: OAuthProvider,
    redirect_uri: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get OAuth login URL for the specified provider.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        redirect_uri: Optional custom redirect URI
        
    Returns:
        Dictionary with login URL
    """
    login_url = await get_oauth_login_url(provider, redirect_uri)
    return {"login_url": login_url}


@router.post("/token/{provider}", response_model=Token)
async def get_token(
    provider: OAuthProvider,
    auth_request: OAuthRequest
) -> Token:
    """
    Exchange OAuth code for access token.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        auth_request: Request with authorization code and redirect URI
        
    Returns:
        Token object
    """
    # Exchange code for OAuth token
    token_info = await get_oauth_token(
        provider, 
        auth_request.code, 
        auth_request.redirect_uri
    )
    
    # Get user info from OAuth provider
    user_info = await get_oauth_user_info(provider, token_info)
    
    # Create JWT access token
    access_token_expires = timedelta(minutes=oauth_config.jwt_expiration_minutes)
    access_token = await create_access_token(
        data={
            "sub": user_info.provider_user_id,
            "provider": user_info.provider,
            "email": user_info.email
        },
        expires_delta=access_token_expires,
    )
    
    # Return token information
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=oauth_config.jwt_expiration_minutes * 60,
        refresh_token=token_info.get("refresh_token")
    )


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: OAuthProvider,
    code: str,
    state: Optional[str] = None,
    error: Optional[str] = None
) -> RedirectResponse:
    """
    Handle OAuth callback from provider.
    
    Args:
        provider: OAuth provider (google, naver, kakao)
        code: Authorization code
        state: Optional state parameter
        error: Optional error parameter
        
    Returns:
        Redirect to frontend with token or error
    """
    if error:
        # Redirect to frontend with error
        return RedirectResponse(
            url=f"/auth/error?error={error}&provider={provider}"
        )
    
    # This endpoint should redirect to the frontend with the code
    # The frontend will then exchange the code for a token using the /token endpoint
    return RedirectResponse(
        url=f"/auth/success?code={code}&provider={provider}&state={state or ''}"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(token_data: TokenData = Depends(get_token_data)) -> UserResponse:
    """
    Get current user information from token.
    
    Args:
        token_data: Token data from JWT
        
    Returns:
        User information
    """
    # In a real application, you would retrieve the user from a database
    # For this example, we'll return a mock user based on the token data
    
    # This is where you would normally query the database
    # user = await get_user_by_id_and_provider(token_data.sub, token_data.provider)
    
    # Mock user response for demonstration purposes
    return UserResponse(
        id=token_data.sub,
        email="user@example.com",  # This would come from your database
        name="User Name",  # This would come from your database
        provider=token_data.provider,
        profile_image=None,
        created_at=token_data.exp or datetime.utcnow(),
        updated_at=None
    ) 