import logging
import os
import sys
from app.components.path_utils import get_project_root

# Adds the parent of the current script's folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def get_logger(name: str = "app_logger", log_file: str = "app.log"):
    """
    Configure and return a logger.
    Logs are both streamed to console and saved to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("\033[32m%(asctime)s | %(levelname)s | %(message)s\033[0m")
        console_handler.setFormatter(console_format)

        # Resolve path to the assets folder from any location
        project_root = get_project_root()
        log_path = os.path.join(project_root, "assets", log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_format)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
