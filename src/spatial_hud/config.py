import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "invert_lr": False,
    "invert_fb": False,
    "rotation": 0.0,
}

class ConfigManager:
    @staticmethod
    def load() -> dict[str, Any]:
        if not CONFIG_FILE.exists():
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                # Merge with defaults to ensure all keys exist
                config = DEFAULT_CONFIG.copy()
                config.update(data)
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return DEFAULT_CONFIG.copy()

    @staticmethod
    def save(data: dict[str, Any]) -> None:
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=4)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
