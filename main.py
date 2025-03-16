import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from routers import auth_routes, chat_routes
from models.auth import TokenData
from utils.auth_utils import get_token_data
from utils.sentiment_analyzer import sentiment_analyzer
from utils.chat_segmenter import ChatSegmenter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context for the application.
    Handles setup and teardown of application resources.
    """
    # Startup: Load resources, initialize connections, etc.
    logger.info("Starting application...")
    
    # NLP Model initialization
    try:
        # The sentiment_analyzer is already initialized as a singleton
        # when imported, but we can log its status
        logger.info("Initializing NLP models for sentiment analysis...")
        logger.info(f"Using sentiment model: {sentiment_analyzer.model_name}")
        
        # Initialize chat segmenter (it will be loaded when used)
        logger.info("Preparing chat segmentation models...")
    except Exception as e:
        logger.error(f"Error initializing NLP models: {str(e)}")
    
    # Here you would initialize resources like:
    # - Database connections
    # - Caching systems
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    
    # Here you would clean up resources like:
    # - Close database connections
    # - Release NLP models
    # - Close cache connections


# Create the FastAPI application with lifespan context
app = FastAPI(
    title="Chat Analysis API",
    description="API for chat conversation analysis and relationship metrics",
    version="0.1.0",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount routers
app.include_router(auth_routes.router)
app.include_router(chat_routes.router)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint, serves the static index.html file"""
    return FileResponse("static/index.html")


@app.get("/hello/{name}")
async def say_hello(name: str):
    """Sample greeting endpoint"""
    return {"message": f"Hello {name}"}


class ProtectedMessage(BaseModel):
    message: str


@app.get("/protected", response_model=ProtectedMessage)
async def protected_route(token_data: TokenData = Depends(get_token_data)):
    """Protected route example requiring authentication"""
    return ProtectedMessage(
        message=f"Hello authenticated user {token_data.username}"
    )
