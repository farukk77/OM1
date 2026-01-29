import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class HotReloadManager:
    """
    Manages the hot-reload strategy. Determines if a change requires
    a full restart or just a runtime reload.
    """
    # Fields that require a full system restart if changed
    RESTART_REQUIRED_FIELDS = {'llm_provider', 'model_name', 'api_key', 'embedding_model'}

    def __init__(self, current_config: Dict[str, Any]):
        self.current_config = current_config

    def check_changes(self, new_config: Dict[str, Any]) -> str:
        """
        Compares new config with current config.
        Returns: 'RESTART', 'RELOAD', or 'NONE'
        """
        if not new_config:
            return 'NONE'

        # Check critical fields
        for field in self.RESTART_REQUIRED_FIELDS:
            old_val = self.current_config.get(field)
            new_val = new_config.get(field)
            if old_val != new_val:
                logger.warning(f"Critical field changed: {field}. Restart required.")
                return 'RESTART'
        
        # If config is different but no critical fields changed
        if new_config != self.current_config:
            return 'RELOAD'
            
        return 'NONE'

    def update_config(self, new_config: Dict[str, Any]):
        self.current_config = new_config
