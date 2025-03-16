import logging
from typing import Dict, List, Any, Optional
import os
import asyncio
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .vector_db import vector_db_manager

logger = logging.getLogger(__name__)

class RAGChatManager:
    """
    Manages RAG-enhanced chat functionality using the vector database.
    """
    
    def __init__(self, skip_init: bool = False):
        """
        Initialize the RAG chat manager with models and parameters.
        
        Args:
            skip_init: If True, skip LLM initialization (useful for testing)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use a Hugging Face hosted model instead of local file
        self.model_name = os.environ.get("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Store conversation histories by user and chat ID
        self.conversation_memory = {}
        
        # Initialize LLM for chat
        if not skip_init:
            self._initialize_llm()
        else:
            logger.info("Skipping LLM initialization for testing")
            self.tokenizer = None
            self.model = None
            self.llm = None
        
    def _initialize_llm(self):
        """Initialize the language model for chat functionality."""
        try:
            logger.info(f"Initializing LLM model: {self.model_name} on {self.device}")
            
            # Load model and tokenizer from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure model loading based on available hardware
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Add device map for GPU if available
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
    
    async def get_chat_chain(self, user_id: str, chat_id: str) -> Optional[ConversationalRetrievalChain]:
        """
        Get a ConversationalRetrievalChain for a specific user and chat.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat
            
        Returns:
            ConversationalRetrievalChain for RAG-enhanced chat
        """
        # Get the vector store for this user/chat
        vector_store = await vector_db_manager.get_vector_db(user_id, chat_id)
        if not vector_store:
            logger.warning(f"No vector database found for user {user_id}, chat {chat_id}")
            return None
        
        # Create memory for conversation history
        memory_key = f"{user_id}_{chat_id}"
        if memory_key not in self.conversation_memory:
            self.conversation_memory[memory_key] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Create retriever from vector store
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create the conversational chain
        try:
            chain = await asyncio.to_thread(
                ConversationalRetrievalChain.from_llm,
                llm=self.llm,
                retriever=retriever,
                memory=self.conversation_memory[memory_key],
                verbose=True
            )
            return chain
        except Exception as e:
            logger.error(f"Error creating conversation chain: {str(e)}", exc_info=True)
            return None
    
    async def chat_with_segments(
        self,
        user_id: str,
        chat_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Chat with the chat history using RAG to retrieve context.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat
            message: User message for the chat
            
        Returns:
            Dict containing the response and retrieved contexts
        """
        chain = await self.get_chat_chain(user_id, chat_id)
        if not chain:
            return {
                "response": "I couldn't access your chat history. Please make sure you've analyzed a chat file first.",
                "contexts": []
            }
        
        try:
            # Get response from the chain
            result = await asyncio.to_thread(
                chain,
                {"question": message}
            )
            
            # Format response with context
            response = {
                "response": result.get("answer", ""),
                "contexts": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat with segments: {str(e)}", exc_info=True)
            return {
                "response": f"I encountered an error while processing your message: {str(e)}",
                "contexts": []
            }
    
    async def clear_chat_history(self, user_id: str, chat_id: str) -> bool:
        """
        Clear the conversation history for a specific user and chat.
        
        Args:
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat
            
        Returns:
            bool: True if successful, False otherwise
        """
        memory_key = f"{user_id}_{chat_id}"
        if memory_key in self.conversation_memory:
            del self.conversation_memory[memory_key]
            return True
        return False

# Initialize a singleton instance
rag_chat_manager = RAGChatManager(skip_init=os.environ.get("TESTING") == "0") 