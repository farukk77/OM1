import json
import asyncio
import logging
import os
import aiofiles
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AsyncJsonStorage:
    """
    Asynchronous JSON persistence layer.
    Fixes #985 by saving history without blocking the main loop.
    Uses atomic writes (write to temp -> rename) to prevent corruption.
    """
    def __init__(self, file_path: str = "conversation_history.json"):
        self.file_path = file_path
        self._lock = asyncio.Lock()

    async def load(self) -> List[Dict[str, Any]]:
        """Loads history from disk asynchronously."""
        if not os.path.exists(self.file_path):
            return []
        
        try:
            async with aiofiles.open(self.file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                if not content.strip():
                    return []
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            return []

    async def save(self, messages: List[Dict[str, Any]]):
        """Saves history to disk asynchronously (Fire & Forget)."""
        try:
            async with self._lock:
                # 1. Atomic write: Write to temp file first
                temp_file = f"{self.file_path}.tmp"
                async with aiofiles.open(temp_file, mode='w', encoding='utf-8') as f:
                    await f.write(json.dumps(messages, indent=2, ensure_ascii=False))
                
                # 2. Atomic Replace: Fast OS-level switch to prevent corruption
                os.replace(temp_file, self.file_path)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")