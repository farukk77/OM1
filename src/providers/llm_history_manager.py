import logging
import asyncio
import sys
import os
from typing import List, Dict

# --- Import Path Setup ---
# Ensure we can import from the 'src/memory' directory regardless of execution context
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(current_dir)  # Navigate up to 'src' directory
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from memory.async_json_storage import AsyncJsonStorage
except ImportError:
    # Fallback import strategy
    from src.memory.async_json_storage import AsyncJsonStorage

logger = logging.getLogger(__name__)

class LLMHistoryManager:
    """
    Manages chat history using an asynchronous JSON persistence layer.
    
    This implementation solves Issue #985 by ensuring that conversation history
    is persisted to disk without blocking the main event loop. It uses
    a 'fire-and-forget' strategy for saving.
    """

    def __init__(self, system_prompt: str = "", max_history: int = 50):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        
        # Initialize the asynchronous storage handler
        self.storage = AsyncJsonStorage("conversation_history.json")
        
        # Trigger background loading of existing history on startup
        asyncio.create_task(self._load_history())

    async def _load_history(self):
        """Asynchronously loads conversation history from disk."""
        try:
            saved_messages = await self.storage.load()
            if saved_messages:
                self.messages = saved_messages
                logger.info(f"Successfully loaded {len(saved_messages)} messages form disk.")
            elif self.system_prompt:
                self.messages = [{"role": "system", "content": self.system_prompt}]
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            if self.system_prompt:
                self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, role: str, content: str):
        """Adds a new message and triggers a non-blocking save."""
        self.messages.append({"role": role, "content": content})
        
        # Enforce history limit
        if len(self.messages) > self.max_history:
            if self.messages[0].get("role") == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_history):]
            else:
                self.messages = self.messages[-(self.max_history):]

        # Fire-and-forget save (runs in background)
        asyncio.create_task(self.storage.save(self.messages))

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        asyncio.create_task(self.storage.save(self.messages))